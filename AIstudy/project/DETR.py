import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import resnet50
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import os

class ObjectDetection(nn.Module):
    def __init__(self, num_classes, num_queries, hidden_dim=256, nheads=8, num_encoder_layers = 6, num_decoder_layers=6):
        super(ObjectDetection, self).__init__()
        
        # Backbone (ResNet + FPN)
        self.backbone = resnet50(

            weights=ResNet50_Weights.IMAGENET1K_V1
        )
        del self.backbone.fc

        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.num_classes = num_classes

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(num_queries//2, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(num_queries//2, hidden_dim // 2))

        self.conv = nn.Conv2d(3,64,kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(64,64,kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2048,hidden_dim, 1)
        self.conv3 = nn.Conv2d(128,64, 1)
    def forward(self, images, targets=None):
        # 이미지 전처리

        # Backbone feature extraction
        out = self.conv(images)
        out0 = self.conv1(out)

        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        print(x.shape, out0.shape)
        x = torch.cat([x ,out0] ,dim = 1)
        x = self.conv3(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Convert to transformer-compatible format
        h = self.conv2(x)  # (batch_size, hidden_dim, H, W)
        H, W = h.shape[-2:]

        # Positional encoding for encoder
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),  # (H, W, hidden_dim/2)
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),  # (H, W, hidden_dim/2)
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # (H*W, 1, hidden_dim)

        # Flatten spatial dimensions for encoder input
        src = (0.1 * h.flatten(2).permute(2, 0, 1))  # (H*W, batch_size, hidden_dim)

        # Prepare tgt for decoder
        if targets is not None:
            # Convert ground truth boxes to the same format as query_pos
            gt_boxes = targets['boxes']  # (batch_size, num_boxes, 4)
            tgt = self.query_pos.unsqueeze(1).repeat(1, h.shape[0], 1)  # (num_queries, batch_size, hidden_dim)
        else:
            # Default to learnable object queries
            tgt = self.query_pos.unsqueeze(1).repeat(1, h.shape[0], 1)  # (num_queries, batch_size, hidden_dim)

        # Transformer forward pass
        memory = self.transformer.encoder(src + pos)  # Encoder pass
        h = self.transformer.decoder(tgt, memory)  # Decoder pass

        # 예측 결과
        pred_logits = self.linear_class(h.transpose(0, 1))  # (batch_size, num_queries, num_classes + 1)
        pred_boxes = self.linear_bbox(h.transpose(0, 1)).sigmoid()  # (batch_size, num_queries, 4)

        # 원본 크기로 바운딩 박스 복원

        return {
            'pred_logits': pred_logits, 
            'pred_boxes': pred_boxes
        }


import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import resnet50
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import os

class ObjectDetection(nn.Module):
    def __init__(self, num_classes, num_queries, hidden_dim=256, nheads=8, num_encoder_layers = 6, num_decoder_layers=6):
        super(ObjectDetection, self).__init__()
        
        # Backbone (ResNet + FPN)
        self.backbone = resnet50(

            weights=ResNet50_Weights.IMAGENET1K_V1
        )
        del self.backbone.fc

        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.num_classes = num_classes

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(num_queries//2, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(num_queries//2, hidden_dim // 2))

        self.conv = nn.Conv2d(3,64,kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(64,64,kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2048,hidden_dim, 1)

    def forward(self, images, targets=None):
        # 이미지 전처리

        # Backbone feature extraction
        out = self.conv(images)
        out0 = self.conv1(out)

        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Convert to transformer-compatible format
        h = self.conv2(x)  # (batch_size, hidden_dim, H, W)
        H, W = h.shape[-2:]

        # Positional encoding for encoder
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),  # (H, W, hidden_dim/2)
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),  # (H, W, hidden_dim/2)
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # (H*W, 1, hidden_dim)

        # Flatten spatial dimensions for encoder input
        src = (0.1 * h.flatten(2).permute(2, 0, 1))  # (H*W, batch_size, hidden_dim)

        # Prepare tgt for decoder
        if targets is not None:
            # Convert ground truth boxes to the same format as query_pos
            gt_boxes = targets['boxes']  # (batch_size, num_boxes, 4)
            tgt = self.query_pos.unsqueeze(1).repeat(1, h.shape[0], 1)  # (num_queries, batch_size, hidden_dim)
        else:
            # Default to learnable object queries
            tgt = self.query_pos.unsqueeze(1).repeat(1, h.shape[0], 1)  # (num_queries, batch_size, hidden_dim)

        # Transformer forward pass
        memory = self.transformer.encoder(src + pos)  # Encoder pass
        h = self.transformer.decoder(tgt, memory)  # Decoder pass

        # 예측 결과
        pred_logits = self.linear_class(h.transpose(0, 1))  # (batch_size, num_queries, num_classes + 1)
        pred_boxes = self.linear_bbox(h.transpose(0, 1)).sigmoid()  # (batch_size, num_queries, 4)

        # 원본 크기로 바운딩 박스 복원

        return {
            'pred_logits': pred_logits, 
            'pred_boxes': pred_boxes
        }


