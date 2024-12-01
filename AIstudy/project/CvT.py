import logging
import torch
import torch.nn as nn
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import DropPath, trunc_normal_
import numpy as np
from torch.nn import ModuleList
import torch.nn.init as init
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler  # 최신 API 사용
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output
from torchvision.ops import nms, generalized_box_iou, box_convert  # 필요한 함수 임포트
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from pycocotools.cocoeval import COCOeval
import random
import os
from PIL import Image
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import warnings

# **ConvEmbed 클래스**
class ConvEmbed(nn.Module):
    def __init__(
        self, patch_size: tuple = (7,7), in_channels=3, embedding_dim=32,
        stride: tuple = (4,4), padding=(3,3), norm=None
    ):
        super(ConvEmbed, self).__init__()
        self.norm_ = norm
        self.proj = nn.Conv2d(
            in_channels=in_channels, out_channels=embedding_dim,
            kernel_size=patch_size, stride=stride, padding=padding
        )
        if self.norm_:
            self.norm = self.norm_(embedding_dim)

    def forward(self, x):

        x = self.proj(x)
        
        if self.norm_:
            n, c, h, w = x.shape
            x = self.norm(rearrange(x, 'n c h w -> n (h w) c'))
            x = rearrange(x, 'n (h w) c -> n c h w', h=h, w=w)
        return x

# **Attention 클래스 (수정됨)**
class Attention(nn.Module):
    def __init__(
        self, dim_in, dim_out, num_heads, qkv_bias=False, 
        attn_drop=0., proj_drop=0., 
        kernel_size=3, stride_kv=2, stride_q=1,
        padding_kv=1, padding_q=1, with_cls_token=False, **kwargs 
    ):
        super().__init__()
        self.scale = dim_out ** -0.5
        self.conv_proj_q = self._build_projection(
            dim_in, kernel_size, padding_q, stride_q,
        )
        self.conv_proj_k = self._build_projection(
            dim_in, kernel_size, padding_kv, stride_kv,

        )
        self.conv_proj_v = self._build_projection(
            dim_in, kernel_size, padding_kv, stride_kv,

        )
        
        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
        self.with_cls_token = with_cls_token
        self.num_heads = num_heads

    @staticmethod
    def _build_projection(dim_in, kernel_size, padding, stride):
        build_proj = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                in_channels=dim_in, out_channels=dim_in,
                kernel_size=kernel_size, stride=stride,
                padding=padding, bias=False, groups=dim_in
            )),
            ('bn', nn.BatchNorm2d(dim_in)),
            ('rearrange', Rearrange('b c h w -> b (h w) c')),
        ]))
        return build_proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)
        print(q.shape,k.shape,v.shape)
        return q, k, v
        
    def forward(self, x, h,w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x,h,w)

        print('1',x.shape, h, w)
        print(q.shape,k.shape,v.shape)
        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)
        print('1233',x.shape, h, w)
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale 
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        print('121',x.shape, h, w)
        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# **MLP 클래스**
class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None,
        activate_method=nn.GELU, drop=0.
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = activate_method()
        out_features = out_features or in_features
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# **Block 클래스**
class Block(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, qkv_bias=False, attn_drop=0.0, drop=0.0, drop_path=0.0,
                 mlp_ratio=4, activate_method=nn.GELU, norm=nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm(dim_in)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm(dim_out)
        self.mlp = MLP(in_features=dim_out, hidden_features=int(dim_out * mlp_ratio), activate_method=activate_method,
                       drop=drop)

    def forward(self, x, h, w):
        x_ = self.norm1(x)
        attention_ = self.attn(x_, h, w)
        x_ = x + self.drop_path(attention_)
        return x_ + self.drop_path(self.mlp(self.norm2(x_)))

# **LayerNorm_ 클래스**
class LayerNorm_(nn.LayerNorm):
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x_type = x.type()
        x = super().forward(x.type(torch.float32)).type(x_type)
        return x
# **QuickGELU 클래스**
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x*torch.sigmoid(1.702*x)
        return x

# **VisionTransformer 클래스 (수정됨)**
class VisionTransformer(nn.Module):
    def __init__(
        self, patch_size=(16, 16), patch_stride=(16, 16), patch_padding=0,
        in_channels=3, embedding_dim=768, depth=12, num_heads=12, 
        mlp_ratio=4, qkv_bias=False, drop_rate=0.1, attn_drop_rate=0.1,
        drop_path_rate=0.1, activate_method=nn.GELU, norm=nn.LayerNorm, **kwargs
    ):
        super(VisionTransformer, self).__init__()
        self.out_channels = embedding_dim
        self.patch_embed = ConvEmbed(
            patch_size=patch_size, in_channels=in_channels, stride=patch_stride,
            padding=patch_padding, embedding_dim=embedding_dim, norm=norm
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embedding_dim)) if kwargs.get('with_cls_token', False) else None
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.ModuleList([Block(
            dim_in=embedding_dim, dim_out=embedding_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=drop_path, activate_method=activate_method,
            norm=norm, **kwargs) for drop_path in dpr
                                        ])
        self.apply(self.init_weights_trunc_normal)
        
        # Position Embeddings
        num_patches = (1536 // patch_stride[0]) * (2048 // patch_stride[1])  # 14x14=196
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if self.cls_token is not None else 0), embedding_dim))
        trunc_normal_(self.pos_embed, std=0.02)

    @property
    def get_out_channels(self):
        return self.out_channels

    @staticmethod
    def init_weights_trunc_normal(layer):
        if isinstance(layer, nn.Linear):
            trunc_normal_(layer.weight, std=0.02)
            if isinstance(layer.bias, torch.Tensor):
                init.constant_(layer.bias, 0)
        elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm2d)):
            init.constant_(layer.bias, 0)
            init.constant_(layer.weight, 1.0)

    def forward(self, x):
        """input: n c h w => output: n c h w"""
        x = self.patch_embed(x)  # n c h w
        n, c, h, w = x.size()
        x = rearrange(x, 'n c h w -> n (h w) c')  # [n, 196, c]
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(n, -1, -1)  # [n,1,c]
            x = torch.cat((cls_tokens, x), dim=1)  # [n,197,c]
            x = x + self.pos_embed  # [n,197,c]
        else:
            x = x + self.pos_embed[:, :h*w, :]  # [n,196,c]
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, h, w)
        if self.cls_token is not None:
            x = x[:, 1:, :]  # CLS 토큰 제외, [n,196,c]
        x = rearrange(x, 'n (h w) c -> n c h w', h=h, w=w)  # [n,c,14,14]
        return x  # n c h w

# **VisionTransformerWithAnchors 클래스 (수정됨)**
class VisionTransformerWithAnchors(nn.Module):
    def __init__(
        self,
        num_classes,
        patch_size=(16, 16),
        patch_stride=(16, 16),
        patch_padding=0,
        in_channels=3,
        embedding_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        activate_method=QuickGELU,
        norm=LayerNorm_,
        with_cls_token=True  # 수정: cls_token 사용 안함
    ):
        super(VisionTransformerWithAnchors, self).__init__()
        self.num_classes = num_classes
        self.vit = VisionTransformer(
            patch_size=patch_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            activate_method=activate_method,
            norm=norm,
            with_cls_token=with_cls_token  # False로 설정
        )
        
        # Feature Pyramid Network (FPN) 추가
        self.fpn = nn.ModuleList([
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=2, padding=1)
        ])
        
        # AnchorGenerator 초기화 (3개의 피처 맵에 맞춤)
        self.anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 3
        )
        
        # 예측 헤드 추가 (클래스 및 바운딩 박스 예측용)
        num_anchors = len(self.anchor_generator.aspect_ratios[0])
        self.cls_head = nn.Linear(embedding_dim, num_classes * num_anchors)
        self.bbox_head = nn.Linear(embedding_dim, 4 * num_anchors)
    
    def forward(self, x):
        # ViT를 통해 피처 맵 추출
        features = self.vit(x)  # [batch_size, embedding_dim, H, W]

        # FPN을 통해 다중 스케일 피처 맵 생성
        feature_maps = []
        x_fpn = features
        for fpn_layer in self.fpn:
            x_fpn = fpn_layer(x_fpn)
            feature_maps.append(x_fpn)
        
        # 디버깅 출력
        print(f"Number of feature maps: {len(feature_maps)}")
        for idx, fmap in enumerate(feature_maps):
            print(f"Feature map {idx} shape: {fmap.shape}")
        
        # ImageList 생성
        image_sizes = [(fmap.shape[-2], fmap.shape[-1]) for fmap in feature_maps]
        image_list = ImageList(x, image_sizes)
        
        # Anchor 박스 생성
        anchors = self.anchor_generator(image_list, feature_maps)
        
        # 디버깅 출력
        print(f"Number of anchors generated: {len(anchors)}")
        for idx, anchor in enumerate(anchors):
            print(f"Anchors for feature map {idx} shape: {anchor.shape}")
        
        # 예측 수행
        batch_size = x.shape[0]
        cls_logits = []
        bbox_preds = []
        num_anchors_per_location = len(self.anchor_generator.aspect_ratios[0])
        for fmap in feature_maps:
            fmap_flat = fmap.permute(0, 2, 3, 1).contiguous()
            N, H, W, C = fmap_flat.shape
            fmap_flat = fmap_flat.view(N, H * W, C)  # [batch_size, num_patches, embedding_dim]
            cls_logit = self.cls_head(fmap_flat)  # [batch_size, num_patches, num_classes * num_anchors_per_location]
            bbox_pred = self.bbox_head(fmap_flat)  # [batch_size, num_patches, 4 * num_anchors_per_location]
            
            # 앵커 수에 맞게 차원 조정
            cls_logit = cls_logit.view(N, H * W, num_anchors_per_location, self.num_classes)
            bbox_pred = bbox_pred.view(N, H * W, num_anchors_per_location, 4)
            
            cls_logit = cls_logit.reshape(N, -1, self.num_classes)
            bbox_pred = bbox_pred.reshape(N, -1, 4)
            
            cls_logits.append(cls_logit)
            bbox_preds.append(bbox_pred)
        
        cls_logits = torch.cat(cls_logits, dim=1)  # [batch_size, total_num_anchors, num_classes]
        bbox_preds = torch.cat(bbox_preds, dim=1)  # [batch_size, total_num_anchors, 4]
        
        return {
            "cls_logits": cls_logits,  # [batch_size, total_num_anchors, num_classes]
            "bbox_preds": bbox_preds,  # [batch_size, total_num_anchors, 4]
            "anchors": anchors          # list of tensors
        }
