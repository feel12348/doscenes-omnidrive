# Copyright (c) 2024-2025, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia License.
# To view a copy of this license, visit
# https://github.com/NVlabs/OmniDrive/blob/main/LICENSE
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers

这里版本在原始 StreamPETR 基础上：
    * 保留时序 temp_memory 机制
    * 新增多视角 cross-attn（通过 view_dicts 驱动）
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.utils.checkpoint as cp
import warnings

from torch.nn.utils.rnn import pad_sequence

from mmdet.models.utils.builder import TRANSFORMER
from .attention import FlashMHA


class MultiHeadAttentionwDropout(nn.Module):
    """简化版 Multi-head Attention 包一层 Dropout 和残差。

    输入/输出均为 batch_first: [B, L, C]
    """

    def __init__(self, embed_dims, num_heads, dropout, flash_attn):
        super().__init__()
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._dropout = dropout
        self.flash_attn = flash_attn

        if flash_attn:
            # FlashMHA 使用 q/k/v 关键字参数
            self.attn = FlashMHA(
                embed_dims, num_heads, dropout,
                dtype=torch.float16, device='cuda'
            )
        else:
            # 标准 MultiheadAttention，batch_first=True
            self.attn = nn.MultiheadAttention(
                embed_dims, num_heads,
                dropout=dropout,
                batch_first=True
            )
        self.proj_drop = nn.Dropout(dropout)

        self._count = 0

    def forward(
        self,
        query: Tensor,          # [B, Lq, C]
        key: Tensor,            # [B, Lk, C]
        value: Tensor,          # [B, Lk, C]
        query_pos: Optional[Tensor],  # [B, Lq, C]
        key_pos: Optional[Tensor],    # [B, Lk, C]
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,  # [B, Lk], True 表示 padding
    ):
        """Forward for multi-head attention (batch_first).

        Args:
            query, key, value: [B, L, C]
            query_pos, key_pos: [B, L, C] or None
            attn_mask: 传给 nn.MultiheadAttention 的 attn_mask
            key_padding_mask: [B, Lk]，True 表示被 mask（padding）
        """
        
        if query_pos is not None:
            query_w_pos = query + query_pos
        else:
            query_w_pos = query
        if key_pos is not None:
            key_w_pos = key + key_pos
        else:
            key_w_pos = key
        # import torch.distributed as dist
        # dist.barrier()
        # rank = dist.get_rank()
        # if rank == 0:
        #         import pdb; pdb.set_trace()
        # dist.barrier()

        if self.flash_attn:
            # FlashMHA 的 key_padding_mask 语义通常是 True = keep，
            # 而 pytorch MHA 是 True = masked，所以这里做一次取反
            if key_padding_mask is not None:
                kp = ~key_padding_mask
            else:
                kp = None
            out, attn = self.attn(
                q=query_w_pos,
                k=key_w_pos,
                v=value,
                key_padding_mask=kp,
            )
        else:
            out, attn = self.attn(
                query_w_pos,
                key_w_pos,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        out = self.proj_drop(out)
        # 残差
        return out + query, attn


# Feed-forward Network
class FFN(nn.Module):

    def __init__(self, embed_dims, feedforward_dims, dropout):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Linear(embed_dims, feedforward_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dims, embed_dims),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self._layers(x) + x




class PETRTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_dims,
                 dropout=0.1,
                 flash_attn=True,
                 ):
        super().__init__()
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._feedforward_dims = feedforward_dims

        self.transformer_layers = nn.ModuleList()
        # 1. Multi-head Self-attention (带时序 memory)
        self.transformer_layers.append(
            MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, False)
        )
        # 2. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        # 3. Multi-head Cross-attention (多视角 / 单视角)
        self.transformer_layers.append(
            MultiHeadAttentionwDropout(embed_dims, num_heads, dropout, flash_attn)
        )
        # 4. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )
        # 5. Feed-forward Network
        self.transformer_layers.append(
            FFN(embed_dims, feedforward_dims, dropout))
        # 6. LayerNorm
        self.transformer_layers.append(
            nn.LayerNorm(embed_dims)
        )

    def forward(self,
                query: Tensor,       # [B, Q, C]
                key: Tensor,         # [B, K, C]
                query_pos: Tensor,   # [B, Q, C]
                key_pos: Tensor,     # [B, K, C]
                attn_mask: Optional[Tensor],
                temp_memory: Optional[Tensor] = None,  # [B, M, C]
                temp_pos: Optional[Tensor] = None,     # [B, M, C]
                view_dict: Optional[dict] = None,
                ):
        """Forward for transformer decoder layer.

        Args:
            query:      [B, Q, C]
            key:        [B, K, C]   (图像 / BEV memory)
            query_pos:  [B, Q, C]
            key_pos:    [B, K, C]
            attn_mask:  通常 [B, Q, Q] 或 None
            temp_memory:[B, M, C] 时序 memory
            temp_pos:   [B, M, C]
            view_dict:  多视角信息字典，可为 None（退化为单视角）

                约定字段（和 DVPE 版本一致）：
                    - query_indices: List[Tensor], len = B*V, 每个是 1D idx
                    - query_lens:    List[int],    len = B*V
                    - restore_indices: List[Tensor], len = B，每个是还原 idx
                    - view_key:        [B*V, K_max, C]
                    - view_key_pos:    [B*V, K_max, C]
                    - view_key_padding_mask: [B*V, K_max]  (True = padding)
                    - view_query_pos:  [B*V, Q_max, C]
        """
        B, Q, C = query.shape

        # ===== 1. Self-attn + temporal memory =====
        if temp_memory is not None:
            # [B, Q+M, C]
            temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
            temp_pos_all = torch.cat([query_pos, temp_pos], dim=1)
        else:
            temp_key = temp_value = query
            temp_pos_all = query_pos

        # self-attn：在 (query + temporal memory) 上做注意力
        query, attn0 = self.transformer_layers[0](
            query, temp_key, temp_value,
            query_pos, temp_pos_all,
            attn_mask=attn_mask,
            key_padding_mask=None,
        )
        query = self.transformer_layers[1](query)

        # ===== 2. Cross-attn: 单视角 / 多视角 =====
        if view_dict is None:
            # 单视角：直接对 key 做 cross-attn
            query, attn1 = self.transformer_layers[2](
                query, key, key,
                query_pos, key_pos,
                attn_mask=None,
                key_padding_mask=None,
            )
        else:
            # ---------- 多视角逻辑 ----------
            query_indices = view_dict['query_indices']          # list, len=B*V
            query_lens = view_dict['query_lens']                # list, len=B*V
            restore_indices = view_dict['restore_indices']      # list, len=B

            view_key = view_dict['view_key']                    # [B*V, K_max, C]
            view_key_pos = view_dict['view_key_pos']            # [B*V, K_max, C]
            view_query_pos = view_dict['view_query_pos']        # [B*V, Q_max, C]
            view_key_padding_mask = view_dict['view_key_padding_mask']  # [B*V, K_max]

            V = len(query_indices) // B  # 每个 batch 有多少个视角

            # --- 2.1 按视角切 query，拼成 [B*V, Q_max, C] ---
            view_query_list = []
            for b in range(B):
                for v in range(V):
                    qidx = query_indices[b * V + v]    # Tensor, eg. [len_qv]
                    # query[b, qidx] -> [len_qv, C]
                    view_query_list.append(query[b, qidx])

            # [B*V, Q_max, C]
            view_query = pad_sequence(view_query_list, batch_first=True)
            # import torch.distributed as dist
            # dist.barrier()
            # rank = dist.get_rank()
            # if rank == 0:
            #         import pdb; pdb.set_trace()
            # dist.barrier()


            # --- 2.2 在 view 空间做 cross-attn ---
            view_query, attn1 = self.transformer_layers[2](
                view_query,              # [B*V, Q_max, C]
                view_key.transpose(0, 1).contiguous(),                # [B*V, K_max, C]
                view_key.transpose(0, 1).contiguous(),                # value
                view_query_pos.transpose(0, 1).contiguous(),          # [B*V, Q_max, C]
                view_key_pos.transpose(0, 1).contiguous(),            # [B*V, K_max, C]
                attn_mask=None,
                key_padding_mask=view_key_padding_mask,  # [B*V, K_max]
            )

            # --- 2.3 把 view_query 写回原始 query 顺序 ---
            new_query = query.new_zeros(B, Q, C)
            cur = 0
            for b in range(B):
                tmp_list = []
                for v in range(V):
                    qlen = query_lens[b * V + v]
                    tmp_list.append(view_query[cur, :qlen])  # [qlen, C]
                    cur += 1
                # 视角 concat 后的顺序 -> 用 restore_indices[b] 还原成 [Q, C]
                tmp_cat = torch.cat(tmp_list, dim=0)               # [Q, C]（打乱）
                tmp_cat = tmp_cat[restore_indices[b]]              # [Q, C]（还原）
                new_query[b] = tmp_cat

            query = new_query

        # ===== 3. FFN + LN =====
        query = self.transformer_layers[3](query)
        query = self.transformer_layers[4](query)
        query = self.transformer_layers[5](query)

        return query


@TRANSFORMER.register_module()
class PETRTransformerDecoder_new(nn.Module):
    """多层 Decoder 堆叠，支持时序 memory + 多视角 cross-attn"""

    def __init__(self,
                 num_layers,
                 embed_dims,
                 num_heads,
                 feedforward_dims,
                 dropout,
                 with_cp=False,
                 flash_attn=True):
        super().__init__()
        self._num_layers = num_layers
        self._embed_dims = embed_dims
        self._num_heads = num_heads
        self._feedforward_dims = feedforward_dims
        self._dropout = dropout
        self._with_cp = with_cp

        self._layers = nn.ModuleList()
        for _ in range(num_layers):
            self._layers.append(
                PETRTransformerDecoderLayer(
                    embed_dims,
                    num_heads,
                    feedforward_dims,
                    dropout,
                    flash_attn=flash_attn,
                )
            )

    def forward(self,
                query: Tensor,          # [B, Q, C]
                key: Tensor,            # [B, K, C]
                query_pos: Optional[Tensor] = None,   # [B, Q, C]
                key_pos: Optional[Tensor] = None,     # [B, K, C]
                attn_mask: Optional[Tensor] = None,
                temp_memory: Optional[Tensor] = None, # [B, M, C]
                temp_pos: Optional[Tensor] = None,    # [B, M, C]
                view_dicts: Optional[List[dict]] = None,
                ):
        """Forward function for transformer decoder.

        为了兼容原来接口，只是额外多了 view_dicts，可为 None。
        """
        return_val = []
        num_spin = len(view_dicts) if view_dicts is not None else 0

        for layer_idx, layer in enumerate(self._layers):
            cur_view_dict = None
            if view_dicts is not None and num_spin > 0:
                # 和 DVPE 一样，每一层轮流用不同的旋转 state
                cur_view_dict = view_dicts[layer_idx % num_spin]

            if self._with_cp and self.training:
                # def _layer_forward(q, k, q_pos, k_pos, a_mask, t_mem, t_pos):
                #     return layer(
                #         q,
                #         k,
                #         q_pos,
                #         k_pos,
                #         a_mask,
                #         t_mem,
                #         t_pos,
                #         view_dict=cur_view_dict,
                #     )
                # query = cp.checkpoint(
                #     _layer_forward,
                #     query,
                #     key,
                #     query_pos,
                #     key_pos,
                #     attn_mask,
                #     temp_memory,
                #     temp_pos,
                # )
                # query = cp.checkpoint(
                #     layer,
                #     query,
                #     key,
                #     query_pos,
                #     key_pos,
                #     attn_mask,
                #     temp_memory,
                #     temp_pos,
                #     view_dict=cur_view_dict,
                # )
                query = layer(
                    query,
                    key,
                    query_pos,
                    key_pos,
                    attn_mask,
                    temp_memory,
                    temp_pos,
                    view_dict=cur_view_dict,
                )
            else:
                query = layer(
                    query,
                    key,
                    query_pos,
                    key_pos,
                    attn_mask,
                    temp_memory,
                    temp_pos,
                    view_dict=cur_view_dict,
                )
            return_val.append(query)

        # [num_layers, B, Q, C]
        return torch.stack(return_val, dim=0)


@TRANSFORMER.register_module()
class PETRTemporalTransformer_new(nn.Module):
    """顶层封装：保持原有 StreamPETR 接口不变，只是在 forward 里支持 view_dicts。"""

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 query_number=32,
                 num_layers=6,
                 embed_dims=256,
                 num_heads=8,
                 feedforward_dims=2048,
                 dropout=0.0,
                 with_cp=False,
                 flash_attn=True,):

        super().__init__()
        assert output_dimension % embed_dims == 0, \
            "output dimension (language model) must be divisible by the embed dimension"

        # 注意：这里保持你原始代码行为，不去改 input_dimension 的使用方式
        self.input_dimension = embed_dims
        self.output_dimension = output_dimension

        self.query_decoder = PETRTransformerDecoder_new(
            num_layers=num_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_dims=feedforward_dims,
            dropout=dropout,
            with_cp=with_cp,
            flash_attn=flash_attn)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)

    def forward(self,
                query: Tensor,          # [B, Q, C]
                key: Tensor,            # [B, K, C]
                query_pos: Optional[Tensor] = None,   # [B, Q, C]
                key_pos: Optional[Tensor] = None,     # [B, K, C]
                attn_mask: Optional[Tensor] = None,
                temp_memory: Optional[Tensor] = None, # [B, M, C]
                temp_pos: Optional[Tensor] = None,    # [B, M, C]
                view_dicts: Optional[List[dict]] = None,
                ):
        """Forward function for transformer decoder.

        Args:
            query:        [B, num_queries, C]
            key:          [B, num_tokens, C] （图像 / BEV memory）
            query_pos:    [B, num_queries, C]
            key_pos:      [B, num_tokens, C]
            temp_memory:  [B, M, C] 时序 memory
            temp_pos:     [B, M, C]
            view_dicts:   Optional[List[dict]]，多视角信息；如果为 None，则退化为原始 StreamPETR 的单视角。

        Returns:
            out: [num_layers, B, num_queries, C]
        """

        out = self.query_decoder(
            query,
            key,
            query_pos,
            key_pos,
            attn_mask,
            temp_memory,
            temp_pos,
            view_dicts=view_dicts,
        )
        return out
