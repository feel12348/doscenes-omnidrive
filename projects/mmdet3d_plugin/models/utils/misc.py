## Copyright (c) 2024-2025, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia License.
# To view a copy of this license, visit
# https://github.com/NVlabs/OmniDrive/blob/main/LICENSE
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import numpy as np
from mmdet.core import bbox_xyxy_to_cxcywh
from mmdet.models.utils.transformer import inverse_sigmoid
from peft import LoraConfig, get_peft_model
from ..dense_heads.llava_llama import LlavaLlamaForCausalLM
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
import math

def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:]) 
    return memory * prev_exist
    
def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape
        
        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)
        
        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def apply_ltrb(locations, pred_ltrb): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])# x1
        pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])# y1
        pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])# x2
        pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])# y2
        min_xy = pred_boxes[..., 0].new_tensor(0)
        max_xy = pred_boxes[..., 0].new_tensor(1)
        pred_boxes  = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
        pred_boxes  = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
        pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)


        return pred_boxes    

def apply_center_offset(locations, center_offset): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        centers_2d = torch.zeros_like(center_offset)
        locations = inverse_sigmoid(locations)
        centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
        centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
        centers_2d = centers_2d.sigmoid()

        return centers_2d

@torch.no_grad()
def locations(features, stride, pad_h, pad_w):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """

        h, w = features.size()[-2:]
        device = features.device
        
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        
        locations = locations.reshape(h, w, 2)
        
        return locations



def gaussian_2d(shape, sigma=1.0):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

class SELayer_Linear(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
        

class ViewManipulator:
    def __init__(self, num_view, num_spin, init_angle=0):
        self.num_view = num_view
        self.num_spin = num_spin
        self.init_angle = math.radians(init_angle)
        self.aug_angle = 0
        self.fov = 2*math.pi / self.num_view
        self.state_angle = self.fov / self.num_spin
        
    
    def angle_aug(self):
        self.aug_angle = torch.rand(1).item() * self.fov

    def _cut_by_angle(self, coordinates, state):
        angles = torch.atan2(coordinates[:, :, 1], coordinates[:, :, 0])
        rotation_angle = self.init_angle + self.aug_angle + self.state_angle * state
        angles = torch.fmod(angles + rotation_angle, 2 * math.pi)
        groups = torch.floor(angles / self.fov).long() % self.num_view
        return groups

    def cut_batch_view(self, coordinates, state, restore=False):
        '''
        points : (B, N, 3)
        n : int
        '''
        B = coordinates.shape[0]
        groups = self._cut_by_angle(coordinates, state)
        lens = []
        indices = []
        for b in range(B):
            for v in range(self.num_view):
                index = torch.nonzero(groups[b] == v, as_tuple=False).flatten()
                indices.append(index)
                lens.append(index.size(0))
            
        if not restore:
            return indices, lens

        restore_indices = []
        for b in range(B):
            batch_index = torch.cat(indices[b*self.num_view:(b+1)*self.num_view])
            restore_indices.append(torch.argsort(batch_index))
        return indices, lens, restore_indices

    def transform_to_view(self, coords, view_idx, state, inverse=False, trans=True, dim=3):
        rotation_angle = self.init_angle + self.aug_angle + self.state_angle * state
        angle = self.fov*(2*view_idx+1) - 2*rotation_angle
        angle = (math.pi - angle)/2
        if not inverse:
            angle = -angle
        if dim == 3:
            R = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                            [math.sin(angle), math.cos(angle), 0],
                            [0, 0, 1]]).to(coords.device)
        elif dim == 2:
            R = torch.tensor([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]]).to(coords.device)
        elif dim == 1:
            R = torch.tensor([[math.cos(angle), math.sin(angle)],
                            [-math.sin(angle), math.cos(angle)]]).to(coords.device)
        if trans:
            return (coords-0.5)@R + 0.5
        else:
            return coords@R
        
        
class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, with_ln=True, export_onnx=False):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.with_ln = with_ln
        self.export_onnx = export_onnx

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.with_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=export_onnx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

        if self.with_ln and self.export_onnx:
            nn.init.ones_(self.ln.weight)
            nn.init.zeros_(self.ln.bias)
            
    def forward(self, x, c):
        if self.with_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out


def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points

def transform_reference_points_lane(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1).unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points

def _resize_embedding_layer(old_embedding, vocab_size):
    old_vocab_size, embedding_dim = old_embedding.weight.shape
    if vocab_size <= old_vocab_size:
        return old_embedding
    new_embedding = torch.nn.Embedding(
        vocab_size,
        embedding_dim,
        padding_idx=old_embedding.padding_idx,
        device=old_embedding.weight.device,
        dtype=old_embedding.weight.dtype)
    new_embedding.weight.data[:old_vocab_size] = old_embedding.weight.data
    new_embedding.weight.data[old_vocab_size:] = old_embedding.weight.data.mean(
        dim=0, keepdim=True)
    return new_embedding


def _resize_lm_head(old_head, vocab_size):
    old_vocab_size, hidden_size = old_head.weight.shape
    if vocab_size <= old_vocab_size:
        return old_head
    new_head = torch.nn.Linear(
        hidden_size,
        vocab_size,
        bias=old_head.bias is not None,
        device=old_head.weight.device,
        dtype=old_head.weight.dtype)
    new_head.weight.data[:old_vocab_size] = old_head.weight.data
    new_head.weight.data[old_vocab_size:] = old_head.weight.data.mean(
        dim=0, keepdim=True)
    if old_head.bias is not None:
        new_head.bias.data[:old_vocab_size] = old_head.bias.data
        new_head.bias.data[old_vocab_size:] = old_head.bias.data.mean()
    return new_head


def _ensure_token_embeddings(model, vocab_size):
    if vocab_size is None:
        return model
    target_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if vocab_size > target_model.get_input_embeddings().num_embeddings:
        target_model.resize_token_embeddings(vocab_size)
    if vocab_size > target_model.get_input_embeddings().num_embeddings:
        target_model.set_input_embeddings(
            _resize_embedding_layer(target_model.get_input_embeddings(), vocab_size))
    if (hasattr(target_model, 'model')
            and hasattr(target_model.model, 'embed_tokens')
            and vocab_size > target_model.model.embed_tokens.num_embeddings):
        target_model.model.embed_tokens = _resize_embedding_layer(
            target_model.model.embed_tokens, vocab_size)
    if hasattr(target_model, 'lm_head') and vocab_size > target_model.lm_head.out_features:
        target_model.lm_head = _resize_lm_head(target_model.lm_head, vocab_size)
    for owner in (target_model, model):
        if hasattr(owner, 'config'):
            owner.config.vocab_size = max(owner.config.vocab_size, vocab_size)
    return model


def _assert_token_embeddings(model, vocab_size):
    if vocab_size is None:
        return
    target_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    input_size = target_model.get_input_embeddings().num_embeddings
    model_size = None
    if hasattr(target_model, 'model') and hasattr(target_model.model, 'embed_tokens'):
        model_size = target_model.model.embed_tokens.num_embeddings
    lm_size = target_model.lm_head.out_features if hasattr(target_model, 'lm_head') else None
    if (input_size < vocab_size
            or (model_size is not None and model_size < vocab_size)
            or (lm_size is not None and lm_size < vocab_size)):
        raise RuntimeError(
            f"LLM token embeddings were not resized: vocab_size={vocab_size}, "
            f"input_embeddings={input_size}, model_embed_tokens={model_size}, "
            f"lm_head={lm_size}.")


def load_model(base_model, use_lora, frozen, vocab_size=None, enable_drivecode_numbers=False):
    # import torch.distributed as dist
    # dist.barrier()
    # rank = dist.get_rank()
    # if rank == 0:
    #         import pdb; pdb.set_trace()
    # dist.barrier()
    model = LlavaLlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map='cpu')
    model = _ensure_token_embeddings(model, vocab_size)

    
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        
    # target_modules = []
    # for name, module in model.named_modules():
    #     if any(x in name for x in ("q_proj", "k_proj", "v_proj", "o_proj")) \
    #     and "audio_tower" not in name:
    #         target_modules.append(name.split('.')[-1])  # 这里要按你模型结构稍微处理下
    # import torch.distributed as dist
    # dist.barrier()
    # rank = dist.get_rank()
    # if rank == 0:
    #         import pdb; pdb.set_trace()
    # dist.barrier()
    model.gradient_checkpointing_enable()

    
    
    if use_lora:
        peft_config = LoraConfig(
                r=128,
                lora_alpha=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM")

        model = get_peft_model(model, peft_config)
        model = _ensure_token_embeddings(model, vocab_size)

        # for name, module in model.named_modules():
        #     if "audio_tower" in name and hasattr(module, "lora_A"):
        #         module.lora_A.default.weight.data.zero_()
        #         module.lora_B.default.weight.data.zero_()
        #         module.scaling = 0.0
        #         # 可选：也别训练它
        #         for p in module.parameters():
        #             p.requires_grad = False
        
        ADAPTER_NAME = "default"
        for name, module in model.named_modules():
            if "audio_tower" in name and isinstance(module, LoraLayer):
                # 不让这条 LoRA 输出任何东西
                if ADAPTER_NAME in module.scaling:
                    module.scaling[ADAPTER_NAME] = 0.0

                # 也别训练
                for p in module.parameters():
                    p.requires_grad = False

        for param in filter(lambda p: p.requires_grad,model.parameters()):
            param.data = param.data.to(torch.float32)

    target_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    for module_name in ['number_projector', 'number_head']:
        module = getattr(target_model, module_name, None)
        if module is not None:
            if enable_drivecode_numbers:
                module.float()
            for param in module.parameters():
                param.requires_grad = bool(enable_drivecode_numbers)
    for module in model.modules():
        module.drivecode_enabled = bool(enable_drivecode_numbers)
        if hasattr(module, 'config'):
            module.config.drivecode_enabled = bool(enable_drivecode_numbers)

    model = _ensure_token_embeddings(model, vocab_size)
    _assert_token_embeddings(model, vocab_size)
               
    return model
