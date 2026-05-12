import random
import torch


class Augmentor:
    def __init__(self, brightness_range, contrast_range):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def random_brightness(self, x):
        factor_range = self.brightness_range
        factor = torch.rand(1).item() * (factor_range[1] - factor_range[0]) + factor_range[0]

        return torch.clamp(x * factor, 0, 1)

    def random_contrast(self, x):
        factor_range = self.contrast_range
        factor = torch.rand(1).item() * (factor_range[1] - factor_range[0]) + factor_range[0]
        mean = torch.mean(x, (1, 2, 3), True)

        return torch.clamp(mean + factor * (x - mean), 0, 1)

    def apply_random_augmentation(self, x):
        data_enhancement_dict = {1: self.random_brightness, 2: self.random_contrast}
        aug_func = random.choice(list(data_enhancement_dict.values()))

        return aug_func(x)