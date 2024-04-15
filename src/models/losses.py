import torch
from torch import nn

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        return {"cross_entropy_loss": self.loss(x, y["label"].float())}
    
class MultiCrossEntropy(nn.Module):
    def __init__(self, modalities):
        super(MultiCrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.modalities = modalities

    def forward(self, x, y):
        """
        Args:
            x: dict that contains m modality: torch.Tensor BxN
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: MultiCrossEntropy loss between x and y: torch.Tensor([B]))
        """
        out = {}
        for modality in self.modalities:
            out['_'.join([modality, 'ce_loss'])] = self.loss(x[modality], y["label"].float())
        return out
    
class BCEWithLogs(nn.Module):
    def __init__(self):
        super(BCEWithLogs, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x, y):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: BCE loss between x and y: torch.Tensor([B])
        """
        return {"bce_loss": self.loss(x.float(), y["label"].float())}
    
class MILNCE(nn.Module):
    def __init__(self, modalities, tau=0.1):
        super(MILNCE, self).__init__()
        self.tau = tau
        self.modalities = modalities
        self.n_modalities = len(modalities)

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, input, y):
        """
        Args:
            input: dict that contains m modality: torch.Tensor BxN_patchesxD
            y: dict that contains "label": torch.Tensor BxN (we do not use it)
        Returns:
            torch.Tensor: MIL-NCE loss where we just exclude the diagonal by block as described in the paper
        """
        x, _ = input
        features = [
            x['_'.join(['tokens', modality])]
            for modality in self.modalities]
        n_patches = features[0].shape[1] 
        n_tokens = features[0].shape[1] * features[0].shape[0]
        features = torch.cat(features, dim=0)
        features = features.flatten(0, 1)
        logits = self.cosine_similarity(features, features, normalize=True)
        rows, cols = logits.size()

        bool = torch.ones((rows, cols), dtype=torch.bool)
        for i in range (len(bool)//n_patches):
            bool[i*n_patches:(i+1)*n_patches, i*n_patches:(i+1)*n_patches] = False

        loss = torch.sum(
            torch.logsumexp(
                logits[bool].view(rows, cols - n_patches) / self.tau,
                dim=1,
            )
        )

        # Get the positive examples
        idx = torch.tensor([[i + j * n_tokens for j in range(self.n_modalities) if not (k == j)] for k in range (self.n_modalities) 
                            for i in range(n_tokens)]).to(features.device)
        pos_logits = torch.gather(logits, 1, idx)

        loss += torch.sum(-torch.logsumexp(pos_logits / self.tau, dim=1))
        return {
            "contrastive_loss": loss / len(features),
            "logits": logits
        }
    
class MILNCE_nointra(nn.Module):
    def __init__(self, modalities, tau=0.1):
        super(MILNCE_nointra, self).__init__()
        self.tau = tau
        self.modalities = modalities
        self.n_modalities = len(modalities)

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, input, y):
        """
        Args:
            input: dict that contains m modality: torch.Tensor BxN_patchesxD
            y: dict that contains "label": torch.Tensor BxN (we do not use it)
        Returns:
            torch.Tensor: MIL-NCE loss where we exclude intra-modality
        """
        x, _ = input
        features = [
            x['_'.join(['tokens', modality])]
            for modality in self.modalities]
        n_tokens = features[0].shape[1] * features[0].shape[0]
        features = torch.cat(features, dim=0)
        features = features.flatten(0, 1)
        logits = self.cosine_similarity(features, features, normalize=True)
        rows, cols = logits.size()

        bool = torch.ones((rows, cols), dtype=torch.bool)
        for i in range (self.n_modalities):
            bool[i*n_tokens:(i+1)*n_tokens, i*n_tokens:(i+1)*n_tokens] = False

        loss = torch.sum(
            torch.logsumexp(
                logits[bool].view(rows, cols - n_tokens) / self.tau,
                dim=1,
            )
        )

        # Get the positive examples
        idx = torch.tensor([[i + j * n_tokens for j in range(self.n_modalities) if not (k == j)] for k in range (self.n_modalities) 
                                         for i in range(n_tokens)]).to(features.device)
        pos_logits = torch.gather(logits, 1, idx)

        loss += torch.sum(-torch.logsumexp(pos_logits / self.tau, dim=1))
        return {
            "contrastive_loss": loss / len(features),
            "logits": logits
        }
    
class MILNCE_naive(nn.Module):
    def __init__(self, modalities, tau=0.1):
        super(MILNCE_naive, self).__init__()
        self.tau = tau
        self.modalities = modalities
        self.n_modalities = len(modalities)

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, input, y):
        """
        Args:
            input: dict that contains m modality: torch.Tensor BxN_patchesxD
            y: dict that contains "label": torch.Tensor BxN (we do not use it)
        Returns:
            torch.Tensor: MIL-NCE loss where we exclude just the diagonal
        """
        x, _ = input
        features = [
            x['_'.join(['tokens', modality])]
            for modality in self.modalities]
        n_patches = features[0].shape[1] 
        n_tokens = features[0].shape[1] * features[0].shape[0]
        features = torch.cat(features, dim=0)
        features = features.flatten(0, 1)
        logits = self.cosine_similarity(features, features, normalize=True)
        rows, cols = logits.size()
        indices = torch.arange(0, rows, device=features.device)
        loss = torch.sum(
            torch.logsumexp(
                logits[indices != indices.view(-1, 1)].view(rows, cols - 1) / self.tau,
                dim=1,    
            )
        )
        # Get the positive examples
        idx = torch.tensor([[i + j * n_tokens for j in range(self.n_modalities) if not (k == j)] for k in range (self.n_modalities) 
                            for i in range(n_tokens)]).to(features.device)
        pos_logits = torch.gather(logits, 1, idx)

        loss += torch.sum(-torch.logsumexp(pos_logits / self.tau, dim=1))
        return {
            "contrastive_loss": loss / len(features),
            "logits": logits
        }
    
class MILNCE_easy(nn.Module):
    def __init__(self, modalities, tau=0.1):
        super(MILNCE_easy, self).__init__()
        self.tau = tau
        self.modalities = modalities
        self.n_modalities = len(modalities)

    def cosine_similarity(self, a, b, normalize=True):
        if normalize:
            w1 = a.norm(p=2, dim=1, keepdim=True)
            w2 = b.norm(p=2, dim=1, keepdim=True)
            sim_matrix = torch.mm(a, b.t()) / (w1 * w2.t()).clamp(min=1e-8)
        else:
            sim_matrix = torch.mm(a, b.t())
        return sim_matrix

    def forward(self, input, y):
        """
        Args:
            input: dict that contains m modality: torch.Tensor BxN_patchesxD
            y: dict that contains "label": torch.Tensor BxN (we do not use it)
        Returns:
            torch.Tensor: MIL-NCE loss where we exclude all tokens that do not belong to the same patch
        """
        x, _ = input
        features = [
            #torch.cat(torch.distributed.nn.all_gather(
            x['_'.join(['tokens', modality])]
            #), dim=0) 
            for modality in self.modalities]
        n_tokens = features[0].shape[1] * features[0].shape[0]
        features = torch.cat(features, dim=0)
        features = features.flatten(0, 1)
        logits = self.cosine_similarity(features, features, normalize=True)
        rows, cols = logits.size()

        n_patches = 36
        bool = torch.ones((rows, cols), dtype=torch.bool)
        for i in range (n_tokens//n_patches):
            for j in range (self.n_modalities):
                for h in range (self.n_modalities):
                    bool[(i)*n_patches + j * n_tokens:(i+ 1)*n_patches + j * n_tokens, i * n_patches + h * n_tokens:(i + 1)*n_patches + h * n_tokens] = False

        loss = torch.sum(
            torch.logsumexp(
                logits[bool].view(rows, cols - n_patches * self.n_modalities) / self.tau,
                dim=1,
            )
        )

        # Get the positive examples
        idx = torch.tensor([[i + j * n_tokens for j in range(self.n_modalities) if not (k == j)] for k in range (self.n_modalities) 
                                         for i in range(n_tokens)]).to(features.device)
        pos_logits = torch.gather(logits, 1, idx)

        loss += torch.sum(-torch.logsumexp(pos_logits / self.tau, dim=1))
        return {
            "contrastive_loss": loss / len(features),
            "logits": logits
        }
    
class ReconstructionLossAerial(nn.Module):
    def __init__(self, patch_size = 50):
        super(ReconstructionLossAerial, self).__init__()
        self.patch_size = patch_size

    def patchify(self, x):
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.flatten(2, 3)
        x = torch.permute(x,(0,2,1,3,4))
        return x

    def forward(self, pred, y):
        """
        Args:
            pred: torch.Tensor BxN_patchesxC
            y: ground truth tensor BxTxHxWxC
        Returns:
            torch.Tensor: Reconstruction loss for sentinel data
        """
        target = self.patchify(y)
        mean = target.mean(dim=(2, 3, 4), keepdim=True)
        var = target.var(dim=(2, 3, 4), keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean()
        return loss
        

class ReconstructionMaskLossSentinel(nn.Module):
    def __init__(self):
        super(ReconstructionMaskLossSentinel, self).__init__()

    def forward(self, x, y, patches_masked):
        """
        Args:
            x: tuple containing:
                - pred: torch.Tensor B'xN_patchesxC
                - mask: contains the information of we dates we recontructed
            y: ground truth tensor BxTxHxWxC
            patches_masked: patches we masked 
        Returns:
            torch.Tensor: Reconstruction loss for sentinel data
        """
        pred, mask = x
        target = y[mask[:, 0], mask[:, 1]]
        patches_masked = patches_masked[mask[:, 0], :]
        target = target.view(target.shape[0], target.shape[1], -1).permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)

        mean = target.mean(dim=(-2, -1), keepdim=True)
        var = target.var(dim=(-2, -1), keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * patches_masked).sum() / patches_masked.sum()  # mean loss on removed patches
        return loss.mean()
    
class ReconstructionMaskLossSentinelPastis(nn.Module):
    def __init__(self):
        super(ReconstructionMaskLossSentinelPastis, self).__init__()
        self.patch_size = 4

    def patchify(self, x):
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.flatten(2, 3)
        x = torch.permute(x,(0,2,1,3,4))
        x = x.flatten(2, 4)
        return x

    def forward(self, x, y, patches_masked):
        """
        Args:
            x: tuple containing:
                - pred: torch.Tensor B'xN_patchesxC
                - mask: contains the information of we dates we recontructed
            y: ground truth tensor BxTxHxWxC
            patches_masked: patches we masked 
        Returns:
            torch.Tensor: Reconstruction loss for sentinel data
        """
        pred, mask = x
        target = y[mask[:, 0], mask[:, 1]]
        target = self.patchify(target)
        patches_masked = patches_masked[mask[:, 0], :]
        target = target.view(target.shape[0], target.shape[1], -1)
        pred = pred.permute(0, 2, 1)

        mean = target.mean(dim=(-2, -1), keepdim=True)
        var = target.var(dim=(-2, -1), keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * patches_masked).sum() / patches_masked.sum()  # mean loss on removed patches
        return loss.mean()
    
class ReconstructionMonoLossSentinel(nn.Module):
    def __init__(self):
        super(ReconstructionMonoLossSentinel, self).__init__()

    def forward(self, x, y, patches_masked):
        """
        Args:
            pred: torch.Tensor BxN_patchesxC
            y: ground truth tensor BxHxWxC
            patches_masked: patches we masked 
        Returns:
            torch.Tensor: Reconstruction loss for sentinel data
        """
        pred = x
        target = y
        target = target.view(target.shape[0], target.shape[1], -1).permute(0, 2, 1)

        mean = target.mean(dim=(-2, -1), keepdim=True)
        var = target.var(dim=(-2, -1), keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return loss.mean()

class MAEReconstructionLoss(nn.Module):
    def __init__(self, modalities, patch_size):
        super(MAEReconstructionLoss, self).__init__()
        modality_to_loss = {"aerial": ReconstructionLossAerial, "s2": ReconstructionMaskLossSentinel, 
                            "s1-asc": ReconstructionMaskLossSentinel, "s1-des": ReconstructionMaskLossSentinel,
                            "s1": ReconstructionMaskLossSentinel, "s1-mono": ReconstructionMonoLossSentinel,
                            "s2-mono": ReconstructionMonoLossSentinel,
                            }
        self.modalities = modalities
        for i in range(len(modalities)):
            if modalities[i] == 'aerial':
                setattr(self, '_'.join(['loss', modalities[i]]), modality_to_loss[modalities[i]](patch_size))
            else:
                setattr(self, '_'.join(['loss', modalities[i]]), modality_to_loss[modalities[i]]())

    def forward(self, x, y):
        recons, mask = x
        out = {}
        n_tokens = mask.shape[1] // len(self.modalities)
        for i, modality in enumerate(self.modalities):
            if modality == 'aerial':
                out['_'.join([modality, 'reconstruction_loss'])] = getattr(self, '_'.join(['loss', modality]))(recons[
                                                                    '_'.join(['reconstruct', modality])], y[modality])
            else:
                out['_'.join([modality, 'reconstruction_loss'])] = getattr(self, '_'.join(['loss', modality]))(recons[
                                '_'.join(['reconstruct', modality])], y[modality], mask[:, i * n_tokens:(i+1) * n_tokens])
        return out
    
class MAEReconstructionLossPastis(nn.Module):
    def __init__(self, modalities, patch_size):
        super(MAEReconstructionLossPastis, self).__init__()
        modality_to_loss = {"aerial": ReconstructionLossAerial, "s2": ReconstructionMaskLossSentinelPastis, 
                            "s1-asc": ReconstructionMaskLossSentinelPastis, "s1-des": ReconstructionMaskLossSentinel,
                            "s1": ReconstructionMaskLossSentinel, "s1-mono": ReconstructionMonoLossSentinel,
                            "s2-mono": ReconstructionMonoLossSentinel,
                            }
        self.modalities = modalities
        for i in range(len(modalities)):
            if modalities[i] == 'aerial':
                setattr(self, '_'.join(['loss', modalities[i]]), modality_to_loss[modalities[i]](patch_size))
            else:
                setattr(self, '_'.join(['loss', modalities[i]]), modality_to_loss[modalities[i]]())

    def forward(self, x, y):
        recons, mask = x
        out = {}
        n_tokens = mask.shape[1] // len(self.modalities)
        for i, modality in enumerate(self.modalities):
            if modality == 'aerial':
                out['_'.join([modality, 'reconstruction_loss'])] = getattr(self, '_'.join(['loss', modality]))(recons[
                                                                    '_'.join(['reconstruct', modality])], y[modality])
            else:
                out['_'.join([modality, 'reconstruction_loss'])] = getattr(self, '_'.join(['loss', modality]))(recons[
                                '_'.join(['reconstruct', modality])], y[modality], mask[:, i * n_tokens:(i+1) * n_tokens])
        return out
    
class MultiCrossEntropy(nn.Module):
    def __init__(self, modalities):
        super(MultiCrossEntropy, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.modalities = modalities

    def forward(self, x, y):
        """
        Args:
            x: dict that contains m modality: torch.Tensor BxN
            y: dict that contains "label": torch.Tensor BxN
        Returns:
            torch.Tensor: CrossEntropy loss between x and y: torch.Tensor([B])
        """
        out = {}
        for modality in self.modalities:
            out['_'.join([modality, 'ce_loss'])] = self.loss(x[modality], y["label"].float())
        return out

LOSSES = {
    "crossentropy": CrossEntropy,
    "multicrossentropy": MultiCrossEntropy,
    "bce": BCEWithLogs,
    "mil-nce": MILNCE,
    "mae-loss": MAEReconstructionLoss,
    "mae-loss_pastis": MAEReconstructionLossPastis,
}
AVERAGE = {False: lambda x: x, True: lambda x: x.mean(dim=-1)}


class Losses(nn.Module):
    """The Losses meta-object that can take a mix of losses."""

    def __init__(self, mix={}, modalities=[], patch_size=50):
        """Initializes the Losses object.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        super(Losses, self).__init__()
        assert len(mix)
        self.init_losses(mix, modalities, patch_size)

    def init_losses(self, mix, modalities, patch_size):
        """Initializes the losses.
        Args:
            mix (dict): dictionary with keys "loss_name" and values weight
        """
        self.loss = {}
        for m, v in mix.items():
            m = m.lower()
            try:
                if m == "mae-loss" or m == "mae-loss_pastis":
                    self.loss[m] = (LOSSES[m](modalities, patch_size), v)
                elif m in ["mil-nce", "multicrossentropy", "multicrossentropy-pastis"]:
                    self.loss[m] = (LOSSES[m](modalities), v)
                else:
                    self.loss[m] = (LOSSES[m](), v)
            except KeyError:
                raise KeyError(f"Loss {m} not found in {LOSSES.keys()}")

    def forward(self, x, y, average=True):
        """Computes the losses.
        Args:
            x: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            y: dict that contains "gps": torch.Tensor Bx2 or "label": torch.Tensor BxN
            average (bool): whether to average the losses or not
        Returns:
            dict: dictionary with losses
        """
        output = {"loss": 0}
        for loss_name, (loss, weight) in self.loss.items():
            loss_output = loss(x, y)
            for k, v in loss_output.items():
                if k.endswith("_loss"):
                    v = AVERAGE[average](v)
                    output["loss"] += weight * v
                output[k] = v
        return output