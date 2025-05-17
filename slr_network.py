import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, TypedDict

import utils
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
from modules.mobilevit.mobilevit_v2 import MobileViTv2
import modules.resnet as resnet


@dataclass
class ModelOutput(TypedDict):
    feat_len: List[int]
    conv_logits: torch.Tensor
    sequence_logits: torch.Tensor
    conv_sents: List[str]
    recognized_sents: List[str]


class Identity(nn.Module):
    def forward(self, x): return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return torch.matmul(x, F.normalize(self.weight, dim=0))


class SLRModel(nn.Module):
    def __init__(
        self,
        num_classes,
        c2d_type,
        conv_type,
        use_bn=False,
        hidden_size=1024,
        gloss_dict=None,
        loss_weights=None,
        weight_norm=True,
        share_classifier=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weights = loss_weights or {}

        # CNN frontend (2D ResNet)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()
        # self.conv2d = MobileViTv2(num_classes)
        # self.conv2d.classifier = Identity()

        # 1D convolution temporal extractor
        self.conv1d = TemporalConv(
            input_size=512,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes,
        )

        # Sequence model (BiLSTM)
        self.temporal_model = BiLSTMLayer(
            rnn_type="LSTM",
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )

        # Classifier
        cls_layer = NormLinear if weight_norm else nn.Linear
        self.classifier = cls_layer(hidden_size, num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        else:
            self.conv1d.fc = cls_layer(hidden_size, num_classes)

        # Decoder
        self.decoder = utils.Decode(gloss_dict, num_classes, search_mode="beam")

        # Losses
        self.loss = {
            "CTCLoss": nn.CTCLoss(reduction="none", zero_infinity=False),
            "distillation": SeqKD(T=8),
        }

    def forward(self, x, len_x, label=None, label_lgt=None) -> ModelOutput:
        # Video shape: (B, T, C, H, W)
        print(x.shape)
        if x.ndim == 5:
            batch, temp, channel, height, width = x.shape
            x = x.view(-1, *x.shape[2:])  # Merge batches: (B * 300, C, H, W)
            x = self.conv2d(x).view(batch, temp, -1).permute(0, 2, 1)  # (B, 512, T)
        # Otherwise: assume already extracted features
        framewise = x

        conv_out = self.conv1d(framewise, len_x)
        x_seq = conv_out["visual_feat"]
        lgt = conv_out["feat_len"]

        seq_out = self.classifier(self.temporal_model(x_seq, lgt)["predictions"])

        if self.training:
            conv_pred = pred = None
        else:
            pred = self.decoder.decode(seq_out, lgt, batch_first=False, probs=False)
            conv_pred = self.decoder.decode(conv_out["conv_logits"], lgt, batch_first=False, probs=False)

        return {
            "feat_len": lgt,
            "conv_logits": conv_out["conv_logits"],
            "sequence_logits": seq_out,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, output: ModelOutput, label, label_lgt):
        total_loss = 0
        feat_len = output["feat_len"]
        label = label.long()
        label_lgt = label_lgt.long()

        for name, weight in self.loss_weights.items():
            if name == "ConvCTC":
                log_probs = output["conv_logits"].log_softmax(-1)
                loss = self.loss["CTCLoss"](log_probs, label, feat_len, label_lgt).mean()
            elif name == "SeqCTC":
                log_probs = output["sequence_logits"].log_softmax(-1)
                loss = self.loss["CTCLoss"](log_probs, label, feat_len, label_lgt).mean()
            elif name == "Dist":
                loss = self.loss["distillation"](
                    output["conv_logits"], output["sequence_logits"].detach(), use_blank=False
                )
            else:
                continue
            total_loss += weight * loss

        return total_loss
