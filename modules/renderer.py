import os
import cv2
import random
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV
)
import torch
from torchvision.transforms import transforms


class NeuralRenderer:
    def __init__(self, device, obj_mesh):

        self.device = device

        # load 3D mesh with default textures in '3Dmodels/texture.png', the texture will be reload latter
        self.verts, self.faces, self.aux = load_obj(
            obj_mesh,
            device=device,
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=4,
            texture_wrap="repeat"
        )

        # center the mesh at origin
        verts_center = self.verts.mean(0)
        new_center = torch.tensor([0, 0, 0], device=device)
        translation = new_center - verts_center
        self.verts = self.verts + translation

        # standardized object orientation: rotation transforms at the origin
        angle = torch.tensor([270.0]) * (torch.pi / 180)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotation_matrix = torch.tensor([[1, 0, 0], [0, cos_angle, -sin_angle], [0, sin_angle, cos_angle]],
                                       device=device).squeeze()
        self.verts = torch.mm(self.verts, rotation_matrix.T)

        # move back to original position
        self.verts = self.verts - translation

        # configure rendering settings
        self.raster_settings = RasterizationSettings(
            image_size=640,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )

    def render(self, texture, total_size, npz_list, dataset_dir, seed=None):
        # prepare texture for rendering
        if texture.ndimension() == 3:
            texture = texture.unsqueeze(0)

        # create UV texture mapping
        textures = TexturesUV(
            maps=texture,
            faces_uvs=[self.faces.textures_idx],
            verts_uvs=[self.aux.verts_uvs]
        )

        # create mesh with vertices, faces and textures
        capsule_mesh = Meshes(
            verts=[self.verts],
            faces=[self.faces.verts_idx],
            textures=textures
        )

        # select random samples from dataset
        if seed is not None:
            random.seed(seed)
        random_idx = random.sample(range(0, len(npz_list)), total_size)

        # start rendering
        file_number_list = []
        img_batch = []
        for idx in range(total_size):
            # load background image and camera parameters
            file_number = random_idx[idx]
            data = np.load(os.path.join(dataset_dir, f'data{npz_list[file_number]}.npz'))
            file_number_list.append(npz_list[file_number])

            # preprocess background image
            bg = data['img']
            bg = cv2.resize(bg, (640, 640))
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            transform = transforms.ToTensor()
            tensor_bg = transform(bg).permute(1, 2, 0).to(self.device)

            # extract camera and vehicle transformation parameters
            x, y, z = data['cam_trans'][0]
            elev, azim, _ = data['cam_trans'][1]
            _, offset, _ = data['veh_trans'][1]

            # set camera view transformation
            R, T = look_at_view_transform(dist=np.sqrt(x ** 2 + y ** 2 + z ** 2) * 1.5,
                                          elev=-1 * elev,
                                          azim=offset - azim - 90)
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)

            # initialize renderer components
            rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=self.raster_settings
            )

            shader = SoftPhongShader(device=self.device, cameras=cameras)
            renderer = MeshRenderer(rasterizer, shader)

            # render mesh and composite with background
            image = renderer(capsule_mesh)  # (1, 640, 640, 4)
            img = image[0, ..., :3]

            # create mask and blend with background
            mask = (img[:, :, :] > (250 / 255)).int()
            img_final = img * (1 - mask) + tensor_bg * mask

            img_batch.append(img_final.permute(2, 0, 1))

        return img_batch, file_number_list