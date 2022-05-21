import os
import torch
import random
import numpy as np


class PatchPool:
    def __init__(self, pool_size=500):
        assert pool_size > 0
        self.pool_size = pool_size
        self.patches = []
        self.labels = []

    def save_patch_label_pairs(self, path):
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        patche_label_pairs = [(patch.cpu(), label.cpu()) for patch, label in zip(self.patches, self.labels)]
        torch.save(patche_label_pairs, path)

    def load_patch_label_pairs(self, path, device):
        patche_label_pairs = torch.load(path, map_location="cpu")
        self.patches = [patche_label_pair[0].to(device) for patche_label_pair in patche_label_pairs]
        self.labels = [patche_label_pair[1].to(device) for patche_label_pair in patche_label_pairs]

    @torch.no_grad()
    def update(self, patch, label):
        patch = patch.detach().clone()
        if len(self.patches) < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
            self.patches.append(patch)
            self.labels.append(label)
        else:
            del self.patches[0]
            self.patches = self.patches[1:]
            self.patches.append(patch)
            del self.labels[0]
            self.labels = self.labels[1:]
            self.labels.append(label)
            torch.cuda.empty_cache()

    def get_patch_and_label(self):
        idx = int(np.random.uniform(0, 1.0) * len(self.patches))
        return self.patches[idx], self.labels[idx]
