import math
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer import PatchEmbed,TransformerContainer
import torchshow as ts
from einops import rearrange, reduce, repeat
from fast_pytorch_kmeans import KMeans
from fairscale.nn.checkpoint import checkpoint_wrapper
from fvcore.common.registry import Registry
from torch.distributed.algorithms.ddp_comm_hooks import (
    default as comm_hooks_default,
)
from torch.nn.init import trunc_normal_

import utils.checkpoint as cu
import utils.logging as logging
from flow.flow_viz import flow_to_image
from flow.unimatch import UniMatch
from models import head_helper, stem_helper  # noqa
from models.build import MODEL_REGISTRY
from models.common import TwoStreamFusion
from models.duplex_attention import MultiScaleBlock
from models.reversible_mvit import ReversibleMViT
from models.utils import (
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
)
from soft_target_cross_entropy import SoftTargetCrossEntropyLoss

# from ./FlowFormer-Official.configs.submission import get_cfg as get_submission_cfg

logger = logging.get_logger(__name__)

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = ""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
                cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
                cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    print(name)
    print(name)
    print(name)
    model = MODEL_REGISTRY.get(name)(cfg)

    if cfg.BN.NORM_TYPE == "sync_batchnorm_apex":
        try:
            import apex
        except ImportError:
            raise ImportError("APEX is required for this model, pelase install")

        logger.info("Converting BN layers to Apex SyncBN")
        process_group = apex.parallel.create_syncbn_process_group(
            group_size=cfg.BN.NUM_SYNC_DEVICES
        )
        model = apex.parallel.convert_syncbn_model(
            model, process_group=process_group
        )

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cur_device],
            output_device=cur_device,
            find_unused_parameters=True
            if cfg.MODEL.DETACH_FINAL_FC
               or cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            else False,
        )
        if cfg.MODEL.FP16_ALLREDUCE:
            model.register_comm_hook(
                state=None, hook=comm_hooks_default.fp16_compress_hook
            )
    return model


@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Model builder for MViTv1 and MViTv2.

    "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"
    Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2112.01526
    "Multiscale Vision Transformers"
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.enable_detection = cfg.DETECTION.ENABLE
        self.enable_rev = cfg.MVIT.REV.ENABLE
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        self.H = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        self.W = cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE
        head_init_scale = cfg.MVIT.HEAD_INIT_SCALE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.use_fixed_sincos_pos = cfg.MVIT.USE_FIXED_SINCOS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )
        self.patch_embed2 = PatchEmbed(
            img_size=224,
            patch_size=16,
            in_channels=3,
            embed_dims=768,
            tube_size=2,
            conv_type='Conv3d')
        ####flow embedding
        self.flow_patch_embed = stem_helper.PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )
        self.clu_atten = TransformerContainer(
            num_transformer_layers=2,
            embed_dims=197,
            num_heads=1,
            num_frames=16,
            norm_layer=norm_layer,
            hidden_channels=768 * 4,
            operator_order=['self_attn', 'ffn'],
            name="spatial")
        self.temporal_transformer = TransformerContainer(
            num_transformer_layers=2,
            embed_dims=768,
            num_heads=4,
            num_frames=16,
            norm_layer=norm_layer,
            hidden_channels=768 * 4,
            operator_order=['self_attn', 'ffn'],
            name="spatial")
        self.clu_linear = nn.Linear(1576, 768)
        self.drop_after_pos = nn.Dropout(p=0.)
        if cfg.MODEL.ACT_CHECKPOINT:
            self.patch_embed = checkpoint_wrapper(self.patch_embed)
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                                                     1:
                                                     ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                                                           i
                                                       ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims

        if self.enable_rev:

            # rev does not allow cls token
            assert not self.cls_embed_on

            self.rev_backbone = ReversibleMViT(cfg, self)

            embed_dim = round_width(
                embed_dim, dim_mul.prod(), divisor=num_heads
            )

            self.fuse = TwoStreamFusion(
                cfg.MVIT.REV.RESPATH_FUSE, dim=2 * embed_dim
            )

            if "concat" in self.cfg.MVIT.REV.RESPATH_FUSE:
                self.norm = norm_layer(2 * embed_dim)
            else:
                self.norm = norm_layer(embed_dim)

        else:

            self.blocks = nn.ModuleList()

            for i in range(depth):
                num_heads = round_width(num_heads, head_mul[i])
                if cfg.MVIT.DIM_MUL_IN_ATT:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i],
                        divisor=round_width(num_heads, head_mul[i]),
                    )
                else:
                    dim_out = round_width(
                        embed_dim,
                        dim_mul[i + 1],
                        divisor=round_width(num_heads, head_mul[i + 1]),
                    )
                attention_block = MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    input_size=input_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                    pool_first=pool_first,
                    rel_pos_spatial=self.rel_pos_spatial,
                    rel_pos_temporal=self.rel_pos_temporal,
                    rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                    residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                    dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                    separate_qkv=cfg.MVIT.SEPARATE_QKV,
                    clu_depth=True if i == 21 else False
                )

                if cfg.MODEL.ACT_CHECKPOINT:
                    attention_block = checkpoint_wrapper(attention_block)
                self.blocks.append(attention_block)
                if len(stride_q[i]) > 0:
                    input_size = [
                        size // stride
                        for size, stride in zip(input_size, stride_q[i])
                    ]

                embed_dim = dim_out

            self.norm = norm_layer(embed_dim)

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[embed_dim],
                num_classes=num_classes,
                pool_size=[[temporal_size // self.patch_stride[0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.TransformerBasicHead(
                2 * embed_dim
                if ("concat" in cfg.MVIT.REV.RESPATH_FUSE and self.enable_rev)
                else embed_dim,
                num_classes,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                cfg=cfg,
            )
        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.head.projection.weight.data.mul_(head_init_scale)
        self.head.projection.bias.data.mul_(head_init_scale)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    def _forward_reversible(self, x):
        """
        Reversible specific code for forward computation.
        """
        # rev does not support cls token or detection
        assert not self.cls_embed_on
        assert not self.enable_detection

        x = self.rev_backbone(x)

        if self.use_mean_pooling:
            x = self.fuse(x)
            x = x.mean(1)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = self.fuse(x)
            x = x.mean(1)

        x = self.head(x)

        return x

    def prepare_tokens(self, x):
        # Tokenize
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches+1, 768,device='cuda'))
        b = x.shape[0]
        x = self.patch_embed2(x)

        cls_tokens = nn.Parameter(torch.zeros(1, 1, 768,device='cuda'))
        cls_tokens.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        cls_tokens = repeat(cls_tokens, 'b ... -> (repeat b) ...', repeat=x.shape[0])

        x = torch.cat((cls_tokens, x), dim=1)

        x = x +pos_embed
        x = self.drop_after_pos(x)

        return x, cls_tokens, b

    def forward(self, x, flow, bboxes=None, return_attn=False):

        clu_x, _,_ = self.prepare_tokens(x)
        c_input = rearrange(clu_x, '(b t) p d -> b t p d', b=x.shape[0])
        c_input = rearrange(c_input, 'b t p d -> b (t p) d')
        kstack = None
        for i in range(c_input.shape[0]):  # 배치수만큼
            kmeans = KMeans(n_clusters=101, mode='euclidean', verbose=1)
            labels = kmeans.fit_predict(c_input[i])  # 768
            labels = labels.unsqueeze(dim=0)
            if i == 0:
                kstack = labels
            else:
                kstack = torch.cat((kstack, labels), dim=0)
        kstack = rearrange(kstack, 'b (t p)  -> b t p', t=8)
        k_attn = self.clu_atten(kstack.float())
        k_attn = rearrange(k_attn, 'b t p  -> b (t p)')
        k_out = self.clu_linear(k_attn)
        k_out = k_out.unsqueeze(dim=1)



        x, bcthw = self.patch_embed(x)

        flow, bchw = self.flow_patch_embed(flow)  ##

        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        ########### flow
        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            flow += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            flow = torch.cat((cls_tokens, flow), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                flow += self._get_pos_embed(pos_embed, bcthw)
            else:
                flow += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            flow = self.pos_drop(x)

        if self.norm_stem:
            flow = self.norm_stem(x)

        thw = [T, H, W]
        ##############

        if self.enable_rev:
            x = self._forward_reversible(x)

        else:
            for blk in self.blocks:
                x, flow, thw = blk(x, flow, thw)

            x = torch.cat((x, k_out), dim=1)
            x = self.temporal_transformer(x)

            if self.enable_detection:
                assert not self.enable_rev

                x = self.norm(x)
                if self.cls_embed_on:
                    x = x[:, 1:]

                B, _, C = x.shape
                x = x.transpose(1, 2).reshape(B, C, thw[0], thw[1], thw[2])

                x = self.head([x], bboxes)

            else:
                if self.use_mean_pooling:
                    if self.cls_embed_on:
                        x = x[:, 1:]
                    x = x.mean(1)
                    x = self.norm(x)
                elif self.cls_embed_on:
                    x = self.norm(x)
                    x = x[:, 0]
                else:  # this is default, [norm->mean]
                    x = self.norm(x)
                    x = x.mean(1)
                x = self.head(x)

        return x


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class ExtractFlows():
    def __init__(self):
        self.opticalNet = UniMatch(feature_channels=128,
                                   num_scales=2,
                                   upsample_factor=4,
                                   num_head=1,
                                   ffn_dim_expansion=4,
                                   num_transformer_layers=6,
                                   reg_refine=True,
                                   task='flow')
        checkpoint = torch.load(
            "/home/hong/workspace/source/new_vit/flow_pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth",
            map_location='cuda:0')
        self.opticalNet.load_state_dict(checkpoint['model'], strict=True)
        self.opticalNet.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.opticalNet.eval()
        # for k, v in self.opticalNet.named_parameters():
        #     v.requires_grad = False

    def extract_flows(self, imgs):
        opticalflows_list_train = []
        B, _, _, _, _ = imgs.shape
        for j in range(0, 15):
            image1 = imgs[:, :, j, :, :]
            image2 = imgs[:, :, j + 1, :, :]

            opticalflows = self.opticalNet(image1, image2,
                                           attn_type='swin',
                                           attn_splits_list=[2, 8],
                                           corr_radius_list=[-1, 4],
                                           prop_radius_list=[-1, 1],
                                           num_reg_refine=6,
                                           task='flow',
                                           pred_bidir_flow=False,
                                           )

            flow_pr = opticalflows['flow_preds'][-1]

            flow_pr = flow_pr.permute(0, 2, 3, 1)
            flow_pr = flow_pr.cpu().detach().numpy()
            b, c, h, w = flow_pr.shape

            for k in range(b):
                f = flow_to_image(flow_pr[k])

                f = rearrange(f, ' h w c ->  c h w')
                opticalflows_list_train.append(f)
            if j == 14:
                for k in range(b):
                    f = flow_to_image(flow_pr[k])

                    f = rearrange(f, ' h w c ->  c h w')
                    opticalflows_list_train.append(f)
        opticalflows_list_train = torch.tensor(np.array(opticalflows_list_train)).float().to('cuda')
        opticalflows_list_train = opticalflows_list_train.clone().detach().requires_grad_(True)
        flow_results = opticalflows_list_train.view(B, 16, 3, 224, 224)
        flow_results = flow_results.permute(0, 2, 1, 3, 4)
        return flow_results


class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, num_class, lr, args, cfg,max_iter):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(cfg)
        self.max_iter = max_iter
        if cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
            logger.info("Load from given checkpoint file.")
            checkpoint_epoch = cu.load_checkpoint(
                cfg.TRAIN.CHECKPOINT_FILE_PATH,
                self.model,
                cfg.NUM_GPUS > 1,
                inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
                convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
                epoch_reset=cfg.TRAIN.CHECKPOINT_EPOCH_RESET,
                clear_name_pattern=cfg.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN,
                image_init=cfg.TRAIN.CHECKPOINT_IN_INIT,
            )

        self.SoftTargetCrossEntropyLoss = SoftTargetCrossEntropyLoss()
        self.data_name = args.data_name
        # self.get_flow = ExtractFlows()
        self.get_flow = ExtractFlows()
        print("max iter : ",self.max_iter)
    def forward(self, x):
        return self.model(x["video"])

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=5, max_iters=self.max_iter)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs = batch["video"]
        B, _, _, _, _ = imgs.shape
        labels = batch["label"]
        flow_imgs = self.get_flow.extract_flows((imgs))
        imgs = torch.autograd.Variable(imgs, requires_grad=True)
        flow_imgs = torch.autograd.Variable(flow_imgs, requires_grad=True)

        # vis_imgs = torch.permute(flow_imgs, (0, 2, 1, 3, 4))
        # ts.save(vis_imgs[0], './examples/vis_imgs.jpg')
        # vis_imgs1 = torch.permute(imgs, (0, 2, 1, 3, 4))
        # ts.save(vis_imgs1[0], './examples/vis_imgs1.jpg')

        preds = self.model(imgs, flow_imgs)
        loss = self.SoftTargetCrossEntropyLoss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_loss" % mode, loss, batch_size=B)
        self.log("%s_acc" % mode, acc, batch_size=B)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
