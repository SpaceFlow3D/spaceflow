from typing import *
import torch
import torch.nn as nn
import numpy as np
import copy
from transformers import CLIPTextModel, AutoTokenizer
from torchvision import transforms
import torch.nn.functional as F
import rembg
from PIL import Image
import open3d_pycg as o3d
import os
import time
import imageio
import logging
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import Gaussian
from ..utils import render_utils
# from gui import utils (Directly copied the functions here because of coliding dependencies)
from pathlib import Path
from sklearn.decomposition import PCA
from lib.util.model_revisions import dinov2_hub_repository

log = logging.getLogger(__name__)


def merge_meshes(mesh_list):
    merged = o3d.geometry.TriangleMesh()
    v_offset = 0

    for mesh in mesh_list:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles) + v_offset

        merged.vertices.extend(o3d.utility.Vector3dVector(vertices))
        merged.triangles.extend(o3d.utility.Vector3iVector(triangles))

        v_offset += len(vertices)

    return merged


def voxelgrid_to_open3d(voxels: np.ndarray, threshold=0.5):
    if len(voxels.shape) > 3:
        C, D, H, W = voxels.shape
        flat_feats = voxels.reshape(C, -1).transpose(1,0)
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(flat_feats)
        # Compute feature norm and PCA color std

        # Normalize for RGB
        reduced -= reduced.min(0)
        reduced /= reduced.max(0) + 1e-6

        # Compute norms and color std
        norms = np.linalg.norm(flat_feats, axis=1)
        color_std = np.std(reduced, axis=1)

        # Filter: active voxels with non-trivial color
        mask = (norms > threshold) & (color_std > 1e-3)

        # zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        xx, yy, zz = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        valid_coords = coords[mask]
        valid_colors = reduced[mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_coords.astype(np.float32))
        pcd.colors = o3d.utility.Vector3dVector(valid_colors.astype(np.float32))
    else:
        coords = np.argwhere(voxels > threshold)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
    return pcd


def save_voxelgrid_as_ply(voxels: np.ndarray, filename: str, threshold=0.5):
    pcd = voxelgrid_to_open3d(voxels, threshold)
    o3d.io.write_point_cloud(filename, pcd)


def voxelize_sq_francis(file_name):
    superquadric_mesh = o3d.io.read_triangle_mesh(file_name)
    vertices = np.clip(np.asarray(superquadric_mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    superquadric_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    theta = np.pi / 2
    #superquadric_mesh.rotate(R_x, center=(0, 0, 0))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        superquadric_mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    unique_points = np.unique(vertices, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unique_points)
    o3d.io.write_point_cloud("merged_mesh_voxelized.ply", pcd)

    zeros = np.zeros((unique_points.shape[0], 1))
    unique_points_4d = np.hstack((zeros, unique_points))  # shape [N, 4]
    unique_points_4d_torch = torch.from_numpy(unique_points_4d).to(dtype=torch.int32, device='cpu')
    my_coords_orig = unique_points_4d_torch

    coords_dense = torch.ones(1, 1, 64, 64, 64).to(device='cpu', dtype=torch.float32) * 0.0
    for i in range(my_coords_orig.shape[0]):
      x, y, z = my_coords_orig[i, 1], my_coords_orig[i, 2], my_coords_orig[i, 3]
      coords_dense[0, 0, x, y, z] = 1.0
    return coords_dense


class TrellisTextTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis text-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        text_cond_model (str): The name of the text conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        text_cond_model: str = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_text_cond_model(text_cond_model)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model_transform = None
        if os.environ.get("TRELLIS_EAGER_IMAGE_COND", "").strip().lower() in {"1", "true", "yes", "on"}:
            self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisTextTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisTextTo3DPipeline, TrellisTextTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisTextTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_text_cond_model(args['text_cond_model'])

        new_pipeline.image_cond_model_name = args.get('image_cond_model')
        new_pipeline.image_cond_model_transform = None
        if (
            new_pipeline.image_cond_model_name
            and os.environ.get("TRELLIS_EAGER_IMAGE_COND", "").strip().lower() in {"1", "true", "yes", "on"}
        ):
            new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_text_cond_model(self, name: str):
        """
        Initialize the text conditioning model.
        """
        # load model
        model = CLIPTextModel.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        if not name:
            raise ValueError("No image conditioning model configured for this pipeline")
        if (
            'image_cond_model' in self.models
            and getattr(self, 'image_cond_model_name', None) == name
            and getattr(self, 'image_cond_model_transform', None) is not None
        ):
            return
        dinov2_model = torch.hub.load(dinov2_hub_repository(), name, pretrained=True,
                                     trust_repo=True, skip_validation=True)
        target_device = self.device
        dinov2_model.eval().to(target_device)
        self.models['image_cond_model'] = dinov2_model
        self.image_cond_model_name = name
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def _ensure_image_cond_model(self) -> None:
        if 'image_cond_model' not in self.models or getattr(self, 'image_cond_model_transform', None) is None:
            self._init_image_cond_model(getattr(self, 'image_cond_model_name', None))

    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and all(isinstance(t, str) for t in text), "text must be a list of strings"
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        self._ensure_image_cond_model()
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
            if getattr(self, 'rembg_session', None) is None:
                self.rembg_session = rembg.new_session('u2net')
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_spatial_control(self, spatial_control_path: str) -> torch.Tensor:
        """
        Encode the spatial control.
        """        
        spatial_control = voxelize_sq_francis(spatial_control_path).to(device=self.device) # [1, 1, 64, 64, 64]
        spatial_control_latent = self.models['sparse_structure_encoder'](spatial_control) # [1, 8, 16, 16, 16] 
        # Only for debugging:
        save_voxelgrid_as_ply(spatial_control[0, 0].cpu().numpy(), Path(spatial_control_path).parent / "spatial_control_voxlized.ply")
        save_voxelgrid_as_ply(spatial_control_latent[0].cpu().numpy(), Path(spatial_control_path).parent / "spatial_control_latent.ply")
        return spatial_control_latent

    @torch.no_grad()
    def load_mesh_high_control(self, mesh_path: str) -> torch.Tensor:
        """
        Load the high-control mesh and encode it to latent space.

        Args:
            mesh_path (str): The path to the high-control mesh.

        Returns:
            torch.Tensor: Latent space mask.
        """

        spatial_control = voxelize_sq_francis(mesh_path).to(device=self.device) # [1, 1, 64, 64, 64]
        print(f"Loaded high-control mesh from {mesh_path}, voxelized shape: {spatial_control.shape}, non-zero voxels: {(spatial_control > 0).sum().item()}")

        # Reduce resolution to 16^3
        # spatial_control = F.interpolate(spatial_control, size=(16, 16, 16), mode='trilinear', align_corners=False) # [1, 1, 16, 16, 16]
        spatial_control = F.avg_pool3d(spatial_control, kernel_size=4, stride=4) # [1, 1, 16, 16, 16]

        return spatial_control  

    def get_cond_text(self, prompt: List[str]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            prompt (List[str]): The text prompt.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_text(prompt)
        neg_cond = self.text_cond_model['null_cond']
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def get_cond_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    @torch.no_grad()
    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        vis_output_dir: str = None,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            vis_output_dir (str): The directory to save visualization outputs.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        sample_start = time.perf_counter()
        log.info(
            "Sampling TRELLIS sparse structure: samples=%d, resolution=%d, steps=%s",
            num_samples,
            reso,
            sampler_params.get("steps", "default"),
        )
        ret = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        )
        z_s = ret.samples
        log.info("Sampled TRELLIS sparse structure latent in %.2fs", time.perf_counter() - sample_start)

        # Decode occupancy latent
        decode_start = time.perf_counter()
        decoder = self.models['sparse_structure_decoder']
        occupancy = decoder(z_s)
        coords = torch.argwhere(occupancy > 0)[:, [0, 2, 3, 4]].int()
        log.info(
            "Decoded TRELLIS sparse structure occupancy in %.2fs (%d voxels)",
            time.perf_counter() - decode_start,
            coords.shape[0],
        )

        # Save intermediate voxel structure for visualization
        save_voxelgrid_as_ply(
            occupancy[0, 0].cpu().numpy(), "debug/structure_fm_output.ply"
        )

        if (vis_output_dir is not None) and (len(ret.pred_x_0) > 0):
            video_path = os.path.join(vis_output_dir, 'denoising_evolution.mp4')
            self._render_denoising_evolution(ret.pred_x_0, video_path)
        

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            mesh_start = time.perf_counter()
            ret['mesh'] = self.models['slat_decoder_mesh'](slat)
            log.info("Decoded TRELLIS mesh SLAT in %.2fs", time.perf_counter() - mesh_start)
        if 'gaussian' in formats:
            gaussian_start = time.perf_counter()
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)
            log.info("Decoded TRELLIS gaussian SLAT in %.2fs", time.perf_counter() - gaussian_start)
        if 'radiance_field' in formats:
            rf_start = time.perf_counter()
            ret['radiance_field'] = self.models['slat_decoder_rf'](slat)
            log.info("Decoded TRELLIS radiance-field SLAT in %.2fs", time.perf_counter() - rf_start)
        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        if cond['cond'].shape[-1] == 768:
          flow_model_key = 'slat_flow_model_text'
          flow_model = self.models[flow_model_key]
        else:
          flow_model_key = 'slat_flow_model_image'
          flow_model = self.models[flow_model_key]
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        sample_start = time.perf_counter()
        log.info(
            "Sampling TRELLIS SLAT with %s: coords=%d, steps=%s",
            flow_model_key,
            coords.shape[0],
            sampler_params.get("steps", "default"),
        )
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        log.info("Sampled TRELLIS SLAT in %.2fs", time.perf_counter() - sample_start)

        norm_start = time.perf_counter()
        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        log.info("Applied TRELLIS SLAT normalization in %.2fs", time.perf_counter() - norm_start)
        return slat

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        image: Image.Image = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond_text = self.get_cond_text([prompt])
        torch.manual_seed(seed)
        spatial_control_latent = self.encode_spatial_control(sparse_structure_sampler_params['spatial_control_mesh_path'])
        cond_text = {**cond_text, 'control': spatial_control_latent}  
        coords = self.sample_sparse_structure(cond_text, num_samples, sparse_structure_sampler_params)
        cond_text.pop('control')

        if preprocess_image and image is not None:
          image = self.preprocess_image(image)
          cond_image = self.get_cond_image([image])
          cond = cond_image
        else:
          cond = cond_text


        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)


    @torch.no_grad()
    def gen_structure(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
        """
        cond_text = self.get_cond_text([prompt])
        torch.manual_seed(seed)
        spatial_control_latent = self.encode_spatial_control(sparse_structure_sampler_params['spatial_control_mesh_path'])
        cond_text = {**cond_text, 'control': spatial_control_latent}  
        coords = self.sample_sparse_structure(cond_text, num_samples, sparse_structure_sampler_params)

        return coords

    def gen_structure_v2(
        self,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        vis_output_dir: str = None,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
        """
        conditioning_start = time.perf_counter()
        text_start = time.perf_counter()
        cond_text = self.get_cond_text([prompt])
        log.info(f"Encoded TRELLIS text condition in {time.perf_counter() - text_start:.2f}s")

        torch.manual_seed(seed)
        spatial_start = time.perf_counter()
        spatial_control_latent = self.encode_spatial_control(sparse_structure_sampler_params['spatial_control_mesh_path'])
        log.info(f"Encoded spatial control latent in {time.perf_counter() - spatial_start:.2f}s")

        high_control_spatial_control = None
        low_control_spatial_control = None
        lantent_high_control = None
        if (sparse_structure_sampler_params.get('high_control_spatial_control_mesh_path', None) is not None) and (sparse_structure_sampler_params.get('local_tau_mode', None) == 'guidance'):
            high_start = time.perf_counter()
            high_control_spatial_control = self.encode_spatial_control(sparse_structure_sampler_params['high_control_spatial_control_mesh_path'])
            log.info(f"Encoded high-control spatial latent in {time.perf_counter() - high_start:.2f}s")
        elif (sparse_structure_sampler_params.get('high_control_spatial_control_mesh_path', None) is not None) and (sparse_structure_sampler_params.get('local_tau_mode', None) == 'masking'):
            high_mask_start = time.perf_counter()
            high_control_spatial_control = self.load_mesh_high_control(sparse_structure_sampler_params['high_control_spatial_control_mesh_path'])
            log.info(f"Prepared high-control mask in {time.perf_counter() - high_mask_start:.2f}s")
            # low_control_spatial_control = self.load_mesh_high_control(sparse_structure_sampler_params['spatial_control_mesh_path'])
            print("High control sum:", high_control_spatial_control.sum().item())
            # print("Low control sum:", low_control_spatial_control.sum().item())            
            # low_control_spatial_control = low_control_spatial_control - high_control_spatial_control
            # print("Low control after subtracting high control sum:", low_control_spatial_control.sum().item())
            high_latent_start = time.perf_counter()
            lantent_high_control = self.encode_spatial_control(sparse_structure_sampler_params['high_control_spatial_control_mesh_path'])
            log.info(f"Encoded high-control spatial latent in {time.perf_counter() - high_latent_start:.2f}s")
        elif (sparse_structure_sampler_params.get('low_control_superquadric_mask_path', None) is not None) and (sparse_structure_sampler_params.get('local_tau_mode', None) == 'low_control_mask'):
            low_mask_start = time.perf_counter()
            low_control_spatial_control = self.load_mesh_high_control(sparse_structure_sampler_params['low_control_superquadric_mask_path'])
            log.info(f"Prepared low-control mask in {time.perf_counter() - low_mask_start:.2f}s")
            high_start = time.perf_counter()
            high_control_spatial_control = self.encode_spatial_control(sparse_structure_sampler_params['high_control_spatial_control_mesh_path'])
            log.info(f"Encoded high-control spatial latent in {time.perf_counter() - high_start:.2f}s")

        cond_text = {**cond_text, 'control': spatial_control_latent, 'control_high': high_control_spatial_control, 'control_low_mask': low_control_spatial_control, 'latent_high_control': lantent_high_control}
        log.info(f"Prepared TRELLIS structure conditioning in {time.perf_counter() - conditioning_start:.2f}s")
        coords = self.sample_sparse_structure(cond_text, num_samples, sparse_structure_sampler_params, vis_output_dir=vis_output_dir)

        return coords
    
    def voxelize(self, mesh: o3d.geometry.TriangleMesh) -> torch.Tensor:
        """
        Voxelize a mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to voxelize.
            sha256 (str): The SHA256 hash of the mesh.
            output_dir (str): The output directory.
        """
        vertices = np.asarray(mesh.vertices)
        aabb = np.stack([vertices.min(0), vertices.max(0)])
        center = (aabb[0] + aabb[1]) / 2
        scale = (aabb[1] - aabb[0]).max()
        vertices = (vertices - center) / scale
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
        vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
        return torch.tensor(vertices).int().cuda()

    @torch.no_grad()
    def run_variant(
        self,
        mesh: o3d.geometry.TriangleMesh,
        prompt: str,
        num_samples: int = 1,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Run the pipeline for making variants of an asset.

        Args:
            mesh (o3d.geometry.TriangleMesh): The base mesh.
            prompt (str): The text prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            formats (List[str]): The formats to decode the structured latent to.
        """
        cond = self.get_cond([prompt])
        coords = self.voxelize(mesh)
        coords = torch.cat([
            torch.arange(num_samples).repeat_interleave(coords.shape[0], 0)[:, None].int().cuda(),
            coords.repeat(num_samples, 1)
        ], 1)
        torch.manual_seed(seed)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def _render_denoising_evolution(
        self,
        pred_x_0_list: list,
        output_path: str,
        total_frames: int = 1800,
        resolution: int = 512,
    ) -> None:
        """Render each denoising step's pred_x_0 as a voxel cloud and compile into one video."""
        decoder = self.models['sparse_structure_decoder']
        num_steps = len(pred_x_0_list)
        if num_steps == 0:
            return
        frames_per_step = max(1, total_frames // num_steps)

        all_frames = []
        for step_idx, latent in enumerate(pred_x_0_list):
            occupancy = (decoder(latent) > 0).float()      # (B, 1, 64, 64, 64)
            occupied_idx = torch.argwhere(occupancy[0, 0]) # (M, 3)

            if occupied_idx.shape[0] == 0:
                blank = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                all_frames.extend([blank] * frames_per_step)
                continue

            positions = (occupied_idx.float() + 0.5) / 64.0 - 0.5  # (M, 3) in [-0.5, 0.5]
            n = positions.shape[0]
            gs = Gaussian(aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], sh_degree=0, device='cuda')
            gs.from_xyz(positions)
            identity = torch.zeros(n, 4, device=self.device)
            identity[:, 0] = 1.0
            gs.from_rotation(identity)
            gs.from_scaling(torch.full((n, 3), 1.0 / 128, device=self.device))
            gs.from_opacity(torch.full((n, 1), 0.9, device=self.device))
            gs._features_dc   = torch.zeros(n, 1, 3, device=self.device)
            gs._features_rest = None

            step_frames = render_utils.render_video(
                gs, num_frames=frames_per_step, resolution=resolution, r=2, fov=40,
            )['color']
            all_frames.extend(step_frames)
            print(f"[evolution] step {step_idx+1}/{num_steps}: {n} voxels")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        imageio.mimsave(output_path, all_frames, fps=30)
        print(f"[evolution] saved {output_path}  ({len(all_frames)} frames)")
