from torchmetrics import Metric
from torchmetrics import F1Score
import torch
import os

class MetricsMonoModal(Metric):
    """
    Computes the micro, macro and weighted F1 Score for multi label classification
    Args:
        modalities (list): list of modalities used
        num_classes (int): number of classes
        save_results (bool): if True saves prediction in a csv file 
        get_classes (bool): if True returns the classwise F1 Score
    """

    def __init__(
        self,
        modalities: list = [],
        num_classes: int = 15,
        save_results: bool = False,
        get_classes: bool = False
    ):
        super().__init__()
        self.modality = modalities[0]
        self.get_classes = get_classes
        self.f1 = F1Score(task="multilabel", average = "none", num_labels=num_classes)
        self.f1_micro = F1Score(task="multilabel", average = "micro", num_labels=num_classes)
        self.f1_weighted = F1Score(task="multilabel", average = "weighted", num_labels=num_classes)
        self.save_results = save_results
        if save_results:
            self.results = {}

    def update(self, pred, gt):
        self.f1(pred, gt['label'])
        self.f1_micro(pred, gt['label'])
        self.f1_weighted(pred, gt['label'])
        if self.save_results:
            for i, name in enumerate(gt['name']):
                self.results[name] = list(pred.cpu()[i].numpy())

    def compute(self):
        if self.get_classes:
            f1 = self.f1.compute()
            out = {'F1_Score_macro': sum(f1)/len(f1), 'F1_Score_micro': self.f1_micro.compute(), 'F1_Score_weighted': self.f1_weighted.compute()}
            for i in range(len(f1)):
                out['_'.join(['F1_classe', str(i)])] = f1[i]
            return out
        f1 = self.f1.compute()
        out = {'F1_Score_macro': sum(f1)/len(f1), 'F1_Score_micro': self.f1_micro.compute(), 'F1_Score_weighted': self.f1_weighted.compute()}
        if self.save_results:
            out['results'] = self.results
            return out
        return out

class MetricsMultiModal(Metric):
    """
    Computes the micro, macro and weighted F1 Score for multi label classification with UT&T model
    Args:
        modalities (list): list of modalities used
        num_classes (int): number of classes
        save_results (bool): if True saves prediction in a csv file
        get_modalities (bool): if True returns the F1 Score for sub branch of UT&T
        get_classes (bool): if True returns the classwise F1 Score
    """

    def __init__(
        self,
        modalities: list = [],
        num_classes: int = 15,
        save_results: bool = False,
        get_modalities: bool = False,
        get_classes: bool = False,
    ):
        super().__init__()
        self.modalities = modalities
        self.get_modalities = get_modalities
        self.get_classes = get_classes
        self.f1 = F1Score(task="multilabel", average = "none", num_labels=num_classes - 2)
        self.f1_micro = F1Score(task="multilabel", average = "micro", num_labels=num_classes - 2)
        self.f1_weighted = F1Score(task="multilabel", average = "weighted", num_labels=num_classes - 2)
        if self.get_modalities:
            self.f1_m = {}
            for m in self.modalities:
                self.f1_m[m] = F1Score(task="multilabel", average = "none", num_labels=num_classes).cpu()
        self.save_results = save_results
        if save_results:
            self.results = {}

    def update(self, pred, gt):
        self.f1(pred[self.modalities[0]][:,1:-1], gt['label'][:,1:-1])
        self.f1_micro(pred[self.modalities[0]][:,1:-1], gt['label'][:,1:-1])
        self.f1_weighted(pred[self.modalities[0]][:,1:-1], gt['label'][:,1:-1])
        if self.get_modalities:
            for m in self.modalities:
                self.f1_m[m](pred[m][:,1:-1].cpu(), gt['label'][:,1:-1].cpu())
        if self.save_results:
            for i, name in enumerate(gt['name']):
                self.results[name] = list(pred[self.modalities[0]].cpu()[i].numpy())

    def compute(self):
        f1 = self.f1.compute()
        out = {'F1_Score_macro': sum(f1)/len(f1), 'F1_Score_micro': self.f1_micro.compute(), 'F1_Score_weighted': self.f1_weighted.compute()}
        if self.save_results:
            out['results'] = self.results
        if self.get_modalities:
            for m in self.modalities:
                out['_'.join(['F1_Score', m])] = self.f1_m[m].compute()
        if self.get_classes:
            for i in range(len(f1)):
                out['_'.join(['F1_classe', str(i)])] = f1[i]
        return out

class NoMetrics(Metric):
    """
    Computes no metrics or saves a batch of reconstruction to visualise them
    Args:
        save_reconstructs (bool): if True saves a batch of reconstructions
        modalities (list): list of modalities used
        save_dir (str): where to save reconstructions
    """

    def __init__(
        self,
        save_reconstructs: bool = False,
        modalities: list = [],
        save_dir: str = '',
    ):
        super().__init__()
        self.save_dir = save_dir
        self.save_recons = save_reconstructs
        self.modalities = modalities
        if self.save_recons:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.saves = {}
            for modality in self.modalities:
                self.saves[modality] = []
                self.saves['_'.join(['gt', modality])] = []

    def update(self, pred, gt):
        if self.save_recons:
            recons, _ = pred
            for modality in self.modalities:
                if modality == 'aerial':
                    preds = recons['_'.join(['reconstruct', modality])]
                    preds  = preds.permute(0, 2, 1 ,3, 4)
                    preds  = preds.reshape(preds.shape[0],4,6,6,50,50)
                    preds = preds.permute(0, 1, 2, 4, 3, 5)
                    preds = preds.reshape(preds.shape[0],4,300,300)
                    target = gt[modality][:, :, :300, :300]
                else:
                    preds, mask = recons['_'.join(['reconstruct', modality])]
                    preds = preds.view(-1, preds.shape[1], 6, 6)
                    target = gt[modality][mask[:, 0], mask[:, 1]]
                indice = torch.randint(0, len(preds), (1,)).item()
                self.saves[modality].append(preds[indice])
                self.saves['_'.join(['gt', modality])].append(target[indice])

    def compute(self):
        if self.save_recons:
            for key in self.saves.keys():
                for i, tensor in enumerate(self.saves[key]):
                    torch.save(tensor.cpu(), self.save_dir + key + str(i) + ".pt")
        return {}
    
class MetricsContrastif(Metric):
    """
    Computes metrics for contrastive. Given embeddings for all tokens, we compute the cosine similarity matrix.
    The metric computed is the accuracy of the M -1 minimum distances of each line (except diagonal of course) 
    being the same token across other modalities with M the number of modalities.
    Args:
        modalities (list): list of modalities used
    """

    def __init__(
        self,
        modalities: list = [],
    ):
        super().__init__()
        self.modalities = modalities
        self.n_k = len(self.modalities)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        for i in range(len(modalities)):
            self.add_state(modalities[i], default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, logits):
        size = len(logits) // self.n_k
        labels = torch.arange(size).unsqueeze(1)
        labels = torch.cat([labels + i * len(labels) for i in range(self.n_k)], dim=1)
        labels = torch.cat([labels for _ in range(self.n_k)]).to(logits.device)
        for i in range(self.n_k):
            _, top_indices = torch.topk(logits[i * size:(i + 1) * size], k=self.n_k, dim=1, largest=True)
            self.__dict__[self.modalities[i]] += (torch.sum(torch.tensor([top_indices[i, j] in labels[i] 
                                                for i in range(top_indices.size(0)) for j in range(self.n_k)])) - len(top_indices)) / (self.n_k - 1)
        self.count += len(logits)

    def compute(self):
        dict = {}
        for i in range(len(self.modalities)):
            dict['_'.join(['acc', self.modalities[i]])] = self.__dict__[self.modalities[i]] / self.count
        return dict
    