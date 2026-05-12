import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms


class SDG:
    def __init__(self, device, config, phase):
        self.device = device
        self.texture_size = config['general_train']['texture_size']
        self.pert_size = config['general_train']['perturbation_size']

        self.transform = transforms.ToTensor()
        self.resize = transforms.Resize((self.texture_size, self.texture_size))

        # load UV maps
        self.original1 = self.resize(self.transform(Image.open("3Dmodels/original1.png"))).permute(1, 2, 0).to(device)
        self.original2 = self.resize(self.transform(Image.open("3Dmodels/original2.png"))).permute(1, 2, 0).to(device)
        self.maskUV1 = (self.original1 > (250 / 255)).int()
        self.maskUV2 = (self.original2 > (250 / 255)).int()

        if phase == "phase1":
            self.fusion_weights = config['phase1_train']['fusion_weights']
            self.freqs = config['phase1_train']['freqs']
            self.sigma = config['phase1_train']['sigma']

            c1 = self.HSV_to_RGB(*config['colors']['C1'])
            c2 = self.HSV_to_RGB(*config['colors']['C2'])
            c3 = self.HSV_to_RGB(*config['colors']['C3'])
            self.colors = (c1, c2, c3)
            self.bg_color = self.HSV_to_RGB(*config['colors']['C4'])
            self.pert_color = self.HSV_to_RGB(*config['colors']['C5'])
        elif phase == "phase2":
            self.spe_texture = Image.open(config['phase2_train']['spe_texture']).convert("RGBA")
            self.perturbation = Image.open(config['phase2_train']['perturbation']).convert("RGBA")
            # obtain adv_texture for phase1
            self.phase1_texture = Image.alpha_composite(
                self.spe_texture.resize((self.texture_size, self.texture_size), Image.Resampling.BICUBIC),
                self.perturbation.resize((self.texture_size, self.texture_size), Image.Resampling.BICUBIC)
            )
            self.phase1_texture = self.transform(self.phase1_texture).permute(1, 2, 0).to(device)
            self.phase1_texture = self.phase1_texture[..., :3]

            # obtain the mask of the S_p perturbation for generating the pixel perturbation
            self.perturbation = self.transform(self.perturbation).permute(1, 2, 0).to(device)
            self.mask = self.perturbation[..., 3].unsqueeze(-1)  # [texture_size, texture_size, 1]
            self.mask = self.mask.expand(-1, -1, 3)  # [texture_size, texture_size, 3]
            self.mask = (self.mask != 0) * 1  # [texture_size, texture_size, 3]

    def subgen_spe_texture(self, new_subfolder):
        """Speckle Texture Sub-generator"""

        # generate random gradients
        angles = [torch.tensor(2 * np.pi * np.random.rand(freq + 1, freq + 1),
                               dtype=torch.float32, device=self.device, requires_grad=True) for freq in self.freqs]
        gradients = [torch.dstack((torch.cos(angle), torch.sin(angle))) for angle in angles]

        # generate speckle patterns of different scales
        single_spe_texture = []
        for freq, color, gradient in zip(self.freqs, self.colors, gradients):
            noise = self.generate_perlin_noise(self.texture_size, freq, gradient)
            noise_np = noise.detach().cpu().numpy()
            _, noise_binary = cv2.threshold(noise_np * 255, 128, 255, cv2.THRESH_BINARY)
            filled_noise_binary = self.fill_RGB(noise_binary, *color)
            single_spe_texture.append(filled_noise_binary)

        # alpha-compositing multi-scale speckle patterns
        mix_img = Image.new("RGBA", (self.texture_size, self.texture_size), self.bg_color)
        for img in single_spe_texture:
            mix_img = Image.alpha_composite(mix_img, img)

        mix_img.save(os.path.join(new_subfolder, "speckle_texture.png"))

        return self.transform(mix_img).permute(1, 2, 0).to(self.device)

    def subgen_sta_texture(self, angles):
        """Starry Texture Sub-generator"""

        # generate gradients based on angles
        gradients = [torch.dstack((torch.cos(angle), torch.sin(angle))) for angle in angles]

        # generate perlin noises of different scales
        noises = []
        for freq, gradient in zip(self.freqs, gradients):
            noise = self.generate_perlin_noise(self.pert_size, freq, gradient)
            noise = (noise * 255).float()
            alpha_channel = torch.full((self.pert_size, self.pert_size), 255.0,
                                       dtype=torch.float32, device=self.device)
            noise_RGBA = torch.stack((noise, noise, noise, alpha_channel), dim=-1)
            noises.append(noise_RGBA)

        stacked_noises = torch.stack(noises)

        # obtain a fused noise map
        weights_tensor = torch.tensor(self.fusion_weights, dtype=torch.float32, device=self.device).view(-1, 1, 1, 1)
        result = torch.sum(stacked_noises * weights_tensor, dim=0)
        result_float = result / 255.0  # [size, size, 4]

        # truncation
        mask = result_float <= 0.5
        result_float[mask] = 0.5

        # set the color of the perturbation
        colored_image = torch.zeros((self.pert_size, self.pert_size, 4), dtype=torch.float32, device=self.device)
        colored_image[..., :3] = torch.tensor([self.pert_color[0] / 255,
                                               self.pert_color[1] / 255,
                                               self.pert_color[2] / 255], dtype=torch.float32, device=self.device)

        # adjusted the transparency of the pixel points based on the noise map
        colored_image[..., 3] = result_float[:, :, 0]

        # nonlinear mapping
        thresholds = torch.expm1((1 - result_float[..., 0]) ** self.sigma)
        probabilities = torch.rand(self.pert_size, self.pert_size, device=self.device)

        # remove all pixels that meet the removal condition and apply coloring to the retained pixels
        colored_result = torch.where((probabilities < thresholds).unsqueeze(-1),
                                     torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device),
                                     colored_image)

        # resize the perturbation to match the size of the speckle texture
        colored_result = self.resize(colored_result.permute(2, 0, 1)).permute(1, 2, 0)  # [texture_size, texture_size, 4]

        return colored_result

    def gen_phase1_adv_texture(self, spe_texture, perturbation):
        """Phi1"""

        adv_texture = spe_texture * (1 - perturbation[:, :, 3:4]) + perturbation * perturbation[:, :, 3:4]
        resized_adv_texture = self.resize(adv_texture.permute(2, 0, 1)).permute(1, 2, 0)

        tem_texture = resized_adv_texture[:, :, :3] * self.maskUV1 + self.original1 * (1 - self.maskUV1)
        final_adv_texture = tem_texture * (1 - self.maskUV2) + self.original2 * self.maskUV2  # [texture_size, texture_size, 3]

        return final_adv_texture

    def gen_phase2_adv_texture(self, color_perturbation):
        """Phi2"""

        join_texture = self.resize(color_perturbation.permute(2, 0, 1)).permute(1, 2, 0) + self.phase1_texture
        adv_texture = self.phase1_texture * (1 - self.mask) + torch.clip(join_texture, 0., 1.) * self.mask

        tem_texture = adv_texture * self.maskUV1 + self.original1 * (1 - self.maskUV1)
        final_adv_texture = tem_texture * (1 - self.maskUV2) + self.original2 * self.maskUV2  # [texture_size, texture_size, 3]

        return final_adv_texture

    def generate_perlin_noise(self, size, freq, gradients):
        """Perlin Noise Generator"""

        def f(t):
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

        def lerp(a, b, x):
            return a + x * (b - a)

        def perlin(x, y, gradients):
            x0 = torch.floor(x).to(torch.int64)
            y0 = torch.floor(y).to(torch.int64)
            x1 = x0 + 1
            y1 = y0 + 1

            sx = f(x - x0)
            sy = f(y - y0)

            n00 = gradients[x0 % freq, y0 % freq]
            n10 = gradients[x1 % freq, y0 % freq]
            n01 = gradients[x0 % freq, y1 % freq]
            n11 = gradients[x1 % freq, y1 % freq]

            dx0 = x - x0
            dy0 = y - y0
            dx1 = dx0 - 1
            dy1 = dy0 - 1

            dot00 = dx0 * n00[..., 0] + dy0 * n00[..., 1]
            dot10 = dx1 * n10[..., 0] + dy0 * n10[..., 1]
            dot01 = dx0 * n01[..., 0] + dy1 * n01[..., 1]
            dot11 = dx1 * n11[..., 0] + dy1 * n11[..., 1]

            nx0 = lerp(dot00, dot10, sx)
            nx1 = lerp(dot01, dot11, sx)
            nxy = lerp(nx0, nx1, sy)

            return nxy

        grid = torch.stack(torch.meshgrid(torch.arange(0, size, device=self.device),
                                          torch.arange(0, size, device=self.device)), -1).to(torch.float32) / size * freq

        noise = perlin(grid[..., 0], grid[..., 1], gradients)
        # normalize the noise map
        nor_noise = (noise - noise.min()) / (noise.max() - noise.min())

        return nor_noise

    @staticmethod
    def HSV_to_RGB(h, s, v):
        """turn HSB to RGB"""

        s /= 100
        v /= 100
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        r, g, b = 0, 0, 0

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        elif 300 <= h < 360:
            r, g, b = c, 0, x

        r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255
        return int(r), int(g), int(b)

    @staticmethod
    def fill_RGB(noise, r, g, b):
        """colorization"""

        image = Image.fromarray(noise).convert("RGBA")
        datas = image.getdata()

        new_data = []
        for item in datas:
            if item[0] > 200 and item[1] > 200 and item[2] > 200:
                new_data.append((r, g, b, 255))
            elif item[0] < 50 and item[1] < 50 and item[2] < 50:
                new_data.append((0, 0, 0, 0))
            else:
                new_data.append(item)

        image.putdata(new_data)
        return image