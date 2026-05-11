# ------------------------------------------------------------------------
# StreamPETRHead with:
#   - Temporal Memory Bank
#   - Multi-view Cross Attention (DVPE-style)
#   - DN (denoising) training
#   - VLM queries (num_extra), only for visual-language features
# ------------------------------------------------------------------------

import math
import torch
import torch.nn as nn

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.registry import ATTENTION
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils import NormedLinear

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.models.utils.positional_encoding import (
    pos2posemb3d, pos2posemb1d, nerf_positional_encoding
)
from projects.mmdet3d_plugin.models.utils.misc import (
    MLN, topk_gather, transform_reference_points, memory_refresh,
    SELayer_Linear
)
from ..utils.misc import ViewManipulator  # 与 DVPEHead 相同的工具
from torch.nn.utils.rnn import pad_sequence


@HEADS.register_module()
class StreamPETRHead_new(AnchorFreeHead):
    """
    StreamPETR with:
        - Temporal memory bank
        - Multi-view cross attention (DVPE-style)
        - DN training
        - VLM queries (num_extra): no bbox / loss, only semantic features.

    输入:
        img_feats: (B, N, C, H, W)
        intrinsics: (B, N, 4, 4)
        lidar2img: (B, N, 4, 4)
        ego_pose, ego_pose_inv, timestamp, command, can_bus 等沿用原 StreamPETRHead
    """

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 out_dims=4096,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 num_extra=256,       # VLM queries
                 n_control=11,
                 can_bus_len=2,
                 with_mask=False,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 # loss
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 # train / test cfg
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 # DN 相关
                 scalar=5,
                 noise_scale=0.4,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.5,
                 # 多视角相关
                 use_angle_aug=False,
                 num_cut_view=6,
                 num_spin_view=3,
                 init_angle=0,
                 # 3D 采样射线设置（和 DVPEHead 相同）
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start=1.0,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):

        # ---------------- basic detection config ----------------
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10

        if code_weights is not None:
            self.code_weights = code_weights
        else:
            # (cx, cy, cz, w, l, h, yaw, ?, vx, vy)
            self.code_weights = [1.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0,
                                 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        # cls loss 类别权重
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is StreamPETRHead_new):
            assert isinstance(class_weight, float)
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float)
            class_weight = torch.ones(num_classes + 1) * class_weight
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        # assigner / sampler
        if train_cfg:
            assert 'assigner' in train_cfg
            assigner_cfg = train_cfg['assigner']
            self.assigner = build_assigner(assigner_cfg)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        # 保存主要 config
        self.output_dims = out_dims
        self.n_control = n_control
        self.num_query = num_query            # detection queries 个数
        self.num_extra = num_extra            # 纯 VLM queries 个数
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.with_mask = with_mask
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.can_bus_len = can_bus_len

        # denoising 相关
        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split

        self.act_cfg = transformer.get(
            'act_cfg', dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear

        # 多视角相关
        self.use_angle_aug = use_angle_aug
        self.num_cut_view = num_cut_view
        self.num_spin_view = num_spin_view
        self.vm = ViewManipulator(num_cut_view, num_spin_view, init_angle)

        # 3D ray 位置编码配置
        self.depth_num = depth_num
        self.depth_step = depth_step
        self.LID = LID
        self.depth_start = depth_start


        # 调用 AnchorFreeHead 初始化 (会注册 loss, num_classes 等)
        super(StreamPETRHead_new, self).__init__(
            num_classes, in_channels, init_cfg=init_cfg
        )

        # loss 构建
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        # transformer: 建议用 PETRTemporalTransformer
        self.transformer = build_transformer(transformer)

        # 给 decoder layer 填一些属性 (兼容 PETRTemporalDecoderLayer 的 group_dn 逻辑)
        if hasattr(self.transformer, 'decoder'):
            for layer in self.transformer.decoder.layers:
                layer.extra_num_group = 0        # 不用 group-dn
                layer.extra_num_query = 0
                layer.num_query = self.num_query

        # 代码权重 / 匹配权重
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, dtype=torch.float32),
            requires_grad=False)
        self.match_costs = nn.Parameter(
            torch.tensor(self.match_costs, dtype=torch.float32),
            requires_grad=False)
        self.position_range = nn.Parameter(
            torch.tensor(position_range, dtype=torch.float32),
            requires_grad=False
        )
        # 构建 depth 取样位置 coords_d
        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        # bbox coder + pc_range
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = nn.Parameter(
            torch.tensor(self.bbox_coder.pc_range, dtype=torch.float32),
            requires_grad=False)

        # 初始化层
        self._init_layers()
        self.reset_memory()

    # ------------------------------------------------------------------
    # sub-module 构建
    # ------------------------------------------------------------------
    def _init_layers(self):
        """Initialize all sub-modules."""

        # ---------- cls/reg heads ----------
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)]
        )
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)]
        )

        # backbone feature -> transformer embed
        self.input_projection = nn.Linear(self.in_channels, self.embed_dims)

        # VLM 输出维度
        if self.output_dims is not None:
            self.output_projection = nn.Linear(self.embed_dims, self.output_dims)

        # detection query 的 3D reference points (不含 VLM)
        self.reference_points = nn.Embedding(self.num_query, 3)

        # 用于 temporal propagation 的 pseudo queries 的 reference points
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        # 纯 VLM queries 的 embedding
        self.query_embedding = nn.Embedding(self.num_extra, self.embed_dims)

        # can_bus + ego pose 的 embedding (用于拼到 VLM token 后面)
        if self.output_dims is not None:
            # 74 = canbus(14*self.can_bus_len) + command(??) + ego_pose(4x4)
            # 这里保留原始配置，你可以按需要改
            self.can_bus_embed = nn.Sequential(
                nn.Linear(74, self.embed_dims * 4),
                nn.ReLU(),
                nn.Linear(self.embed_dims * 4, self.output_dims),
            )

        # ---------- 多视角相关 ----------
        # depth_num * 3 (x,y,z)
        self.position_dim = self.depth_num * 3
        self.position_encoder = nn.Sequential(
            nn.Linear(self.position_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )

        # Focal-PETR 风格: 先用 cone 注意力做对齐，再用 SE 对 pe 做条件调节
        self.spatial_alignment = MLN(8)
        self.featurized_pe = SELayer_Linear(self.embed_dims)

        # 视图空间 query pos 编码
        self.view_query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # 时间位置编码 / ego pose 编码
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(156)
            self.ego_pose_memory = MLN(156)

        # query_pos 本身（3D 位置 -> embed），沿用原 StreamPETR 写法：
        # 输入是 nerf_positional_encoding 的输出
        self.query_pos = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def init_weights(self):
        """Initialize weights."""
        # reference points 初始化：xy 用极坐标均匀分布，z uniform
        r = torch.sqrt(torch.rand(self.num_query)) * 0.5
        theta = torch.rand(self.num_query) * 2 * math.pi
        with torch.no_grad():
            self.reference_points.weight.data[:, 0] = r * torch.cos(theta) + 0.5
            self.reference_points.weight.data[:, 1] = r * torch.sin(theta) + 0.5
            self.reference_points.weight.data[:, 2].uniform_(0, 1)

            if self.num_propagated > 0:
                r = torch.sqrt(torch.rand(self.num_propagated)) * 0.5
                theta = torch.rand(self.num_propagated) * 2 * math.pi
                self.pseudo_reference_points.weight.data[:, 0] = r * torch.cos(theta) + 0.5
                self.pseudo_reference_points.weight.data[:, 1] = r * torch.sin(theta) + 0.5
                self.pseudo_reference_points.weight.data[:, 2].uniform_(0, 1)
                self.pseudo_reference_points.weight.requires_grad = False

        # transformer 初始化
        self.transformer.init_weights()

        # sigmoid cls 的 bias init
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    # ------------------------------------------------------------------
    # memory bank / temporal 部分与原版 StreamPETRHead 基本相同
    # ------------------------------------------------------------------
    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None
        self.sample_time = None
        self.memory_canbus = None

    def pre_update_memory(self, data):
        B = data['img_feats'].size(0)
        if self.memory_embedding is None:
            # 第一次：直接全 0 初始化
            self.memory_embedding = data['img_feats'].new_zeros(
                B, self.memory_len, self.embed_dims)
            self.memory_reference_point = data['img_feats'].new_zeros(
                B, self.memory_len, 3)
            self.memory_timestamp = data['img_feats'].new_zeros(
                B, self.memory_len, 1)
            self.memory_egopose = data['img_feats'].new_zeros(
                B, self.memory_len, 4, 4)
            self.memory_velo = data['img_feats'].new_zeros(
                B, self.memory_len, 2)
            self.sample_time = data['timestamp'].new_zeros(B)
            self.memory_canbus = data['img_feats'].new_zeros(
                B, self.can_bus_len, 14)
            x = self.sample_time.to(data['img_feats'].dtype)
        else:
            # 累加 timestamp，用 ego_pose_inv 把历史 memory 映射到当前 ego 坐标
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.sample_time += data['timestamp']
            x = (torch.abs(self.sample_time) < 2.0).to(data['img_feats'].dtype)

            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(
                self.memory_reference_point, data['ego_pose_inv'], reverse=False)

            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)
            self.memory_canbus = memory_refresh(self.memory_canbus[:, :self.can_bus_len], x)
            self.sample_time = data['timestamp'].new_zeros(B)
        # import torch.distributed as dist
        # dist.barrier()
        # rank = dist.get_rank()
        # if rank == 0:
        #         import pdb; pdb.set_trace()
        # dist.barrier()

        # 第一帧：用 pseudo_reference_points 填充前 num_propagated 个 memory slot
        if self.num_propagated > 0:
            pseudo_reference_points = (
                self.pseudo_reference_points.weight *
                (self.pc_range[3:6] - self.pc_range[0:3]) +
                self.pc_range[0:3]
            )
            self.memory_reference_point[:, :self.num_propagated] = (
                self.memory_reference_point[:, :self.num_propagated] +
                (1 - x).view(B, 1, 1) * pseudo_reference_points
            )
            self.memory_egopose[:, :self.num_propagated] = (
                self.memory_egopose[:, :self.num_propagated] +
                (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)
            )

    def post_update_memory(self, data, rec_ego_pose,
                           all_cls_scores, all_bbox_preds,
                           outs_dec, mask_dict, rec_can_bus):
        # 取最后一层 decoder 的预测作为 proposals，更新 memory_bank
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            out_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            out_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)

        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(out_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_canbus = torch.cat([rec_can_bus, self.memory_canbus], dim=1)

        # 把 memory 从当前 ego 坐标映射回世界（或者某个全局）系
        self.memory_reference_point = transform_reference_points(
            self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.sample_time -= data['timestamp']
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose

        return out_memory

    def temporal_alignment(self, query_pos, tgt, reference_points):
        """把当前 frame 的 queries 与 memory_bank 对齐（时间 + ego pose）."""
        B = query_pos.size(0)

        # 把 memory 中的 reference_point 归一到 [0,1]
        temp_reference_point = (
            (self.memory_reference_point - self.pc_range[:3]) /
            (self.pc_range[3:6] - self.pc_range[0:3])
        )
        # memory 的 query pos
        temp_pos = self.query_pos(
            nerf_positional_encoding(
                temp_reference_point.repeat(1, 1, self.n_control)
            )
        )
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(
            B, query_pos.size(1), 1, 1
        )

        if self.with_ego_pos:
            # 当前 ego -> 位置编码
            rec_ego_motion = torch.cat(
                [torch.zeros_like(reference_points[..., :1]),
                 rec_ego_pose[..., :3, :].flatten(-2)],
                dim=-1
            )
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            # import torch.distributed as dist
            # dist.barrier()
            # rank = dist.get_rank()
            # if rank == 0:
            #         import pdb; pdb.set_trace()
            # dist.barrier()
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            # memory ego -> 位置编码
            memory_ego_motion = torch.cat(
                [self.memory_timestamp,
                 self.memory_egopose[..., :3, :].flatten(-2)],
                dim=-1
            ).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)

            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        # 时间 embedding
        query_pos = query_pos + self.time_embedding(
            pos2posemb1d(torch.zeros_like(reference_points[..., :1]))
        )
        temp_pos = temp_pos + self.time_embedding(
            pos2posemb1d(self.memory_timestamp).float()
        )

        # 把 memory 的前 num_propagated 个 slot 拼到 query 末尾，作为 propagated queries
        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat(
                [reference_points, temp_reference_point[:, :self.num_propagated]],
                dim=1
            )
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(
                B, query_pos.shape[1] + self.num_propagated, 1, 1
            )
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    # ------------------------------------------------------------------
    # DN 准备
    # ------------------------------------------------------------------
    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        """
        reference_points: (total_queries, 3)  这里 total_queries = num_extra + num_query
        逻辑基本沿用原 StreamPETRHead，只是 VLM query 放在最前面。
        """
        device = reference_points.device
        if self.training and self.with_dn:
            targets = [
                torch.cat(
                    (m['gt_bboxes_3d']._data.gravity_center,
                     m['gt_bboxes_3d']._data.tensor[:, 3:]),
                    dim=1)
                for m in img_metas
            ]
            labels = [m['gt_labels_3d']._data for m in img_metas]
            known = [(torch.ones_like(t)).to(device) for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)

            known_num = [t.size(0) for t in targets]

            labels = torch.cat(labels)
            boxes = torch.cat(targets)
            batch_idx = torch.cat([
                torch.full((t.size(0),), i, dtype=torch.long, device=device)
                for i, t in enumerate(targets)
            ])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # scalar 份复制 + 加噪声
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += rand_prob * diff * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (
                    (known_bbox_center[..., 0:3] - self.pc_range[0:3]) /
                    (self.pc_range[3:6] - self.pc_range[0:3])
                )
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, dim=1) > self.split
                known_labels[mask] = self.num_classes  # 变成 no-object

            single_pad = int(max(known_num)) if len(known_num) > 0 else 0
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3, device=device)

            # 注意：reference_points 只包含 VLM + detection，不包含 DN padding
            padded_reference_points = torch.cat(
                [padding_bbox.unsqueeze(0).repeat(batch_size, 1, 1),
                 reference_points.unsqueeze(0).repeat(batch_size, 1, 1)],
                dim=1
            )

            if len(known_num):
                map_known_indice = torch.cat(
                    [torch.arange(num, device=device) for num in known_num]
                )
                map_known_indice = torch.cat([
                    map_known_indice + single_pad * i
                    for i in range(self.scalar)
                ]).long()
            else:
                map_known_indice = torch.zeros(0, dtype=torch.long, device=device)

            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = \
                    known_bbox_center.to(device)

            # attn_mask: DN queries 与 match queries 之间的可见性
            tgt_size = pad_size + self.num_extra + self.num_query
            attn_mask = torch.ones(
                tgt_size, tgt_size, device=device, dtype=torch.bool
            )
            attn_mask.fill_(False)

            # match query 不能看到 DN 重构 query
            attn_mask[pad_size:, :pad_size] = True
            # DN queries 之间不能互相看到
            for i in range(self.scalar):
                start = single_pad * i
                end = single_pad * (i + 1)
                attn_mask[start:end, :start] = True
                attn_mask[start:end, end:pad_size] = True

            # temporal attn mask（包含 memory_len + propagated）
            query_size = pad_size + self.num_extra + self.num_query + self.num_propagated
            tgt_size_full = pad_size + self.num_extra + self.num_query + self.memory_len
            if self.with_mask:
                # 如果你需要额外的 mask，可以在这里扩展
                pass
            temporal_attn_mask = torch.ones(
                query_size, tgt_size_full, device=device, dtype=torch.bool
            )
            temporal_attn_mask.fill_(False)
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            mask_dict = dict(
                known_indice=known_indice.long(),
                batch_idx=batch_idx.long(),
                map_known_indice=map_known_indice.long(),
                known_lbs_bboxes=(known_labels, known_bboxs),
                know_idx=know_idx,
                pad_size=pad_size
            )

        else:
            # 推理阶段或不使用 DN
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    # ------------------------------------------------------------------
    # 多视角：把 image tokens + reference_points 切成多个 view 并构造 view_dicts
    # ------------------------------------------------------------------
    def get_image_points(self, data, img_metas):
        """
        利用 intrinsics + lidar2img + 均匀采样的 2D 网格 -> 3D 坐标 (B, LEN, depth_num*3)
        并返回:
            coords3d: (B, LEN, D*3), 归一到 position_range
            intrinsic_for_cone: (B, LEN, 2)
            img_points: (B, LEN, 3) 最后一层的 3D 点 (用来做视图划分)
        """
        eps = 1e-5
        x = data['img_feats']  # (B, N, C, H, W)
        B, N, C, H, W = x.shape
        device = x.device

        # 影像尺寸 (假定所有 camera 一致)
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        # 生成 feature map 上每个 cell 的 2D 坐标 (像素坐标系)
        xs = torch.linspace(0.5, pad_w - 0.5, W, device=device)
        ys = torch.linspace(0.5, pad_h - 0.5, H, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        base = torch.stack([grid_x, grid_y], dim=-1)        # (H, W, 2)
        memory_centers = base.view(1, 1, H, W, 2).repeat(B, N, 1, 1, 1)  # (B, N, H, W, 2)
        memory_centers = memory_centers.view(B * N, H, W, 2)

        BN, H, W, _ = memory_centers.shape
        LEN = N * H * W

        # intrinsics: (B, N, 4, 4) -> 取 fx, fy
        intrinsic = torch.stack(
            [data['intrinsics'][..., 0, 0],
             data['intrinsics'][..., 1, 1]],
            dim=-1
        )                                               # (B, N, 2)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.view(B, N, 1, 1, 2).expand(B, N, H * W, 1, 2)
        intrinsic = intrinsic.reshape(B, LEN, 2)

        D = self.coords_d.shape[0]

        # 构造射线
        memory_centers = memory_centers.view(B, N, H, W, 2).reshape(B, LEN, 1, 2)
        memory_centers = memory_centers.expand(B, LEN, D, 2)
        coords_d = self.coords_d.view(1, 1, D, 1).expand(B, LEN, D, 1)
        coords = torch.cat([memory_centers, coords_d], dim=-1)     # (B, LEN, D, 3)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), dim=-1)
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3],
            torch.ones_like(coords[..., 2:3]) * eps
        )
        coords = coords.unsqueeze(-1)                              # (B, LEN, D, 4, 1)

        img2lidars = data['lidar2img'].inverse()                   # (B, N, 4, 4)
        img2lidars = img2lidars.view(B * N, 4, 4).view(
            B, N, 1, 1, 4, 4
        ).expand(B, N, H * W, D, 4, 4).reshape(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]  # (B, LEN, D, 3)
        coords3d[..., 0:3] = (
            (coords3d[..., 0:3] - self.position_range[0:3]) /
            (self.position_range[3:6] - self.position_range[0:3])
        )
        coords3d = coords3d.reshape(B, LEN, D * 3)

        intrinsic_for_cone = intrinsic
        img_points = coords3d[..., -3:]   # 最后一层 depth 的 3D 点

        return coords3d, intrinsic_for_cone, img_points

    def prepare_for_view(self, data, img_metas, reference_points, memory):
        """
        构造 multi-view cross-attention 所需的 view_dicts 列表。
        每个 state (spin) 下：
            - 对 queries / keys 进行视锥裁剪 (ViewManipulator)
            - 为每个视图构造：
                view_query_pos, view_key, view_key_pos, view_key_padding_mask
        """
        coords3d, intrinsic_for_cone, img_points = self.get_image_points(data, img_metas)
        B = reference_points.size(0)
        V = self.num_cut_view
        S = self.num_spin_view

        if self.training and self.use_angle_aug:
            self.vm.angle_aug()

        view_dicts = []
        for state in range(S):
            query_indices, query_lens, restore_indices = \
                self.vm.cut_batch_view(reference_points.detach() - 0.5,
                                       state, restore=True)
            key_indices, key_lens = \
                self.vm.cut_batch_view(img_points.detach() - 0.5, state)
            key_maxlen = max(key_lens)

            view_dict = dict(
                query_indices=query_indices,
                query_lens=query_lens,
                restore_indices=restore_indices,
                key_indices=key_indices,
                key_lens=key_lens,
            )

            reference_points_list = []
            key_list = []
            coords3d_list = []
            intrinsic_list = []
            view_key_padding_mask = torch.zeros(
                B * V, key_maxlen, dtype=torch.bool, device=reference_points.device
            )

            for b in range(B):
                for v in range(V):
                    qidx = query_indices[b * V + v]
                    kidx = key_indices[b * V + v]
                    klen = key_lens[b * V + v]

                    rp = self.vm.transform_to_view(reference_points[b, qidx], v, state)
                    reference_points_list.append(rp)

                    key_list.append(memory[b, kidx])
                    intrinsic_list.append(intrinsic_for_cone[b, kidx])

                    coord = coords3d[b, kidx].reshape(klen, -1, 3)
                    coord = self.vm.transform_to_view(coord, v, state).reshape(klen, -1)
                    coords3d_list.append(coord)

                    view_key_padding_mask[b * V + v, klen:] = True

            view_reference_points = pad_sequence(reference_points_list, batch_first=False)
            view_key = pad_sequence(key_list, batch_first=False)
            view_coords3d = pad_sequence(coords3d_list, batch_first=False)
            view_intrinsic = pad_sequence(intrinsic_list, batch_first=False)

            # query / key 位置编码
            view_query_pos = self.view_query_embedding(
                pos2posemb3d(inverse_sigmoid(view_reference_points))
            )
            pos_embed = inverse_sigmoid(view_coords3d)
            view_key_pos = self.position_encoder(pos_embed)

            # Focal-PETR 风格 spatial alignment
            view_cone = torch.cat(
                [view_intrinsic, view_coords3d[..., -3:], view_coords3d[..., -90:-87]],
                dim=-1
            )
            view_key = self.spatial_alignment(view_key, view_cone)
            view_key_pos = self.featurized_pe(view_key_pos, view_key)

            view_dict.update(
                view_query_pos=view_query_pos,
                view_key=view_key,
                view_key_pos=view_key_pos,
                view_key_padding_mask=view_key_padding_mask
            )
            view_dicts.append(view_dict)

            # import torch.distributed as dist
            # dist.barrier()
            # rank = dist.get_rank()
            # if rank == 0:
            #         import pdb; pdb.set_trace()
            # dist.barrier()


        return view_dicts

    # ------------------------------------------------------------------
    # forward: 多视角 + 时间 memory + DN + VLM
    # ------------------------------------------------------------------
    def forward(self, img_metas, pos_embed=None, **data):
        """
        Args:
            img_metas: list[dict]
            data:
                img_feats: (B, N, C, H, W)
                intrinsics, lidar2img, ego_pose, ego_pose_inv, timestamp,
                command, can_bus, ...
        Returns:
            outs: dict(all_cls_scores, all_bbox_preds, dn_mask_dict)
            vlm_memory: (B, num_extra + 1, out_dims)    # 最后一层 VLM + can_bus
        """
        # 1. 更新 memory_bank（scene 切换 / 位置映射等）
        self.pre_update_memory(data)

        x = data['img_feats']              # (B, N, C, H, W)
        B, N, C, H, W = x.shape
        num_tokens = N * H * W

        # 2. encoder memory: flatten image tokens
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = self.input_projection(memory)         # (B, num_tokens, embed_dims)

        # 3. reference_points: 先拼 VLM 的占位，然后是 detection queries
        ref_pts_det = self.reference_points.weight      # (num_query, 3)
        vlm_dummy = torch.zeros(self.num_extra, 3, device=ref_pts_det.device)
        reference_points = torch.cat([vlm_dummy, ref_pts_det], dim=0)   # (num_extra+num_query, 3)

        # 4. 准备 DN：会在前面插入 DN padding（pad_size）
        reference_points, attn_mask, mask_dict = \
            self.prepare_for_dn(B, reference_points, img_metas)   # (B, pad+num_extra+num_query, 3)

        # 5. query_pos / tgt 初始化
        query_pos = self.query_pos(
            nerf_positional_encoding(
                reference_points.repeat(1, 1, self.n_control)
            )
        )                                           # (B, total_queries, embed_dims)
        tgt = torch.zeros_like(query_pos)

        # 6. temporal alignment: 把 memory_bank 的 topk proposals 作为 propagated queries
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = \
            self.temporal_alignment(query_pos, tgt, reference_points)

        # 7. VLM queries 的 embedding 写进对应位置（DN padding 之后）
        if mask_dict and mask_dict['pad_size'] > 0:
            pad_size = mask_dict['pad_size']
            tgt[:, pad_size:pad_size + self.num_extra, :] = self.query_embedding.weight.unsqueeze(0)
            query_pos[:, pad_size:pad_size + self.num_extra, :] = 0
        else:
            tgt[:, :self.num_extra, :] = self.query_embedding.weight.unsqueeze(0)
            query_pos[:, :self.num_extra, :] = 0

        # 8. multi-view: 根据 reference_points 和 image tokens 构造 view_dicts
        #    pos_embed 这里不再用（为 None），由 view_key_pos 提供位置信息
        view_dicts = self.prepare_for_view(
            data, img_metas, reference_points, memory
        )
        pos_embed = None
        # import torch.distributed as dist
        # dist.barrier()
        # rank = dist.get_rank()
        # if rank == 0:
        #         import pdb; pdb.set_trace()
        # dist.barrier()

        # 9. transformer decode: PETRTemporalTransformer
        outs_dec = self.transformer(
            tgt,
            memory,
            query_pos,
            pos_embed,
            attn_mask,
            temp_memory,
            temp_pos,
            view_dicts
        )

        # 10. 去掉 VLM + DN 对 reference_points 的影响，只保留 detection 部分用于回归 bbox
        if mask_dict and mask_dict['pad_size'] > 0:
            pad_size = mask_dict['pad_size']
            reference_points = torch.cat(
                [reference_points[:, :pad_size, :],
                 reference_points[:, pad_size + self.num_extra:, :]],
                dim=1
            )
        else:
            reference_points = reference_points[:, self.num_extra:, :]

        outs_dec = torch.nan_to_num(outs_dec)

        # 11. 分离 VLM queries 与 detection queries
        if mask_dict and mask_dict['pad_size'] > 0:
            pad_size = mask_dict['pad_size']
            vlm_memory = outs_dec[-1, :, pad_size:pad_size + self.num_extra, :]
            outs_dec = torch.cat(
                [outs_dec[:, :, :pad_size, :],
                 outs_dec[:, :, pad_size + self.num_extra:, :]],
                dim=2
            )
        else:
            vlm_memory = outs_dec[-1, :, :self.num_extra, :]
            outs_dec = outs_dec[:, :, self.num_extra:, :]

        # 12. detection head: cls / reg
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3

            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        # 把归一化中心坐标映射回真实 3D 坐标
        all_bbox_preds[..., 0:3] = (
            all_bbox_preds[..., 0:3] *
            (self.pc_range[3:6] - self.pc_range[0:3]) +
            self.pc_range[0:3]
        )

        # 13. 把 can_bus + ego pose 聚合到 VLM token 里，得到 final vlm_memory
        rec_can_bus = torch.cat(
            [data['command'].unsqueeze(-1), data['can_bus']], dim=-1
        )                                           # (B, can_bus_len, 14)

        memory_ego_pose = self.memory_egopose.reshape(
            B, -1, self.topk_proposals, 4, 4
        ).flatten(-2)

        if self.output_dims is not None:
            vlm_memory = self.output_projection(vlm_memory)   # (B, num_extra, out_dims)
            can_bus_input = torch.cat(
                [
                    rec_can_bus,                          # (B, can_bus_len,14)
                    self.memory_canbus.flatten(-2),       # (B, can_bus_len*14)
                    memory_ego_pose.mean(-2).flatten(-2)  # (B, 4*4*?)
                ],
                dim=-1
            )
            can_bus_embed = self.can_bus_embed(can_bus_input)     # (B, out_dims)
            vlm_memory = torch.cat(
                [vlm_memory, can_bus_embed.unsqueeze(-2)], dim=-2
            )

        # 14. 更新 memory_bank（使用 detection outputs）
        out_memory = self.post_update_memory(
            data, rec_ego_pose, all_cls_scores, all_bbox_preds,
            outs_dec, mask_dict, rec_can_bus.unsqueeze(-2)
        )

        # 15. 打包输出
        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
            outs = dict(
                all_cls_scores=outputs_class,
                all_bbox_preds=outputs_coord,
                dn_mask_dict=mask_dict
            )
        else:
            outs = dict(
                all_cls_scores=all_cls_scores,
                all_bbox_preds=all_bbox_preds,
                dn_mask_dict=None
            )

        return outs, vlm_memory

    # ------------------------------------------------------------------
    # 下方 loss / get_bboxes 逻辑基本沿用原 StreamPETRHead，不再重复解释
    # ------------------------------------------------------------------
    def prepare_for_loss(self, mask_dict):
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(
                1, 2, 0, 3
            )[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(
                1, 2, 0, 3
            )[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0)

        assign_result = self.assigner.assign(
            bbox_pred, cls_score,
            gt_bboxes, gt_labels,
            gt_bboxes_ignore, self.match_costs,
            self.match_with_velo
        )
        sampling_result = self.sampler.sample(
            assign_result, bbox_pred, gt_bboxes
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_bboxes.new_full(
            (num_bboxes,), self.num_classes, dtype=torch.long
        )
        label_weights = gt_bboxes.new_ones(num_bboxes)

        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)

        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):

        assert gt_bboxes_ignore_list is None
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single,
            cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list,
            gt_bboxes_ignore_list
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):

        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = self.get_targets(
            cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list,
            gt_bboxes_ignore_list
        )

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor
        )

        # regression
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(
            reduce_mean(num_total_pos), min=1
        ).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def dn_loss_single(self,
                       cls_scores,
                       bbox_preds,
                       known_bboxs,
                       known_labels,
                       num_total_pos=None):

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 3.14159 / 6 * \
            self.split * self.split * self.split
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )

        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights,
            avg_factor=cls_avg_factor
        )

        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(
            reduce_mean(num_total_pos), min=1
        ).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    @force_fp32(apply_to=('preds_dicts',))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):

        assert gt_bboxes_ignore is None

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            torch.cat(
                (gt_bboxes.gravity_center,
                 gt_bboxes.tensor[:, 3:]),
                dim=1
            ).to(device) for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list
        )

        loss_dict = dict()
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(
                losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        # DN loss
        if preds_dicts['dn_mask_dict'] is not None:
            (known_labels, known_bboxs,
             output_known_class, output_known_coord,
             num_tgt) = self.prepare_for_loss(preds_dicts['dn_mask_dict'])

            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]

            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single,
                output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list,
                all_num_tgts_list
            )
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]

            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1

        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single,
                all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                all_gt_bboxes_ignore_list
            )
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()
                num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts',))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Decode predicted bboxes."""
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
