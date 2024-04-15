import torch
import random
import torchvision
import torchvision.transforms.v2.functional as TF

class Identity(object):
    def __init__(self):
        pass

    def __call__(self, batch):     
        return batch

class Transform(object):
    def __init__(self, p = 0.0, size = 300):
        self.p = p
        self.crop = torchvision.transforms.v2.RandomCrop(size=[size, size])

    def __call__(self, batch):
        keys = list(batch.keys())
        keys.remove('label')
        keys.remove('name')

        if 'aerial' in keys:
            batch['aerial'] = self.crop(batch['aerial'])

        if random.random() < self.p:
            for key in keys:
                batch[key] = TF.horizontal_flip(batch[key])
        
        if random.random() < self.p:
            for key in keys:
                batch[key] = TF.vertical_flip(batch[key])

        if random.random() < self.p:
            for key in keys:
                batch[key] = TF.rotate(batch[key], 90)
        
        return batch
    
class TransformMAE(object):
    def __init__(self, p = 0.5, size = 224, s2_size = 0):
        self.p = p
        self.s2 = False
        self.crop = torchvision.transforms.Resize(size=[size,size], antialias=True)
        if s2_size > 0:
            self.s2 = True
            self.crop2 = torchvision.transforms.Resize(size=[s2_size, s2_size], antialias=True)

    def __call__(self, batch):
        keys = list(batch.keys())
        keys.remove('label')
        keys.remove('name')

        for key in keys:
            if self.s2 and key in ['s2-4season-median', 's2-median']:
                batch[key] = self.crop2(batch[key])
            else:
                batch[key] = self.crop(batch[key]) 

        return batch
    