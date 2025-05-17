import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, TypedDict
from torchaudio.models import Conformer

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
        window_size: int = 300,
        stride: int = 150,
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
        self.window_size = window_size
        self.stride = stride

        # CNN frontend (2D ResNet)
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()

        # 1D convolution temporal extractor
        self.conv1d = TemporalConv(
            input_size=512,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes,
        )

        # Sequence model (BiLSTM)
        # self.temporal_model = BiLSTMLayer(
        #     # rnn_type="LSTM",
        #     input_size=hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=2,
        #     bidirectional=True,
        # )
        self.temporal_model = Conformer(
            input_dim=hidden_size,
            num_heads=4,
            ffn_dim=hidden_size*4,
            num_layers=12,
            depthwise_conv_kernel_size=31,
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
        # ── 1) Extract per-frame features via sliding‐window through 2D ResNet ──
        if x.ndim == 5:
            # x: (B, T, C, H, W)
            B, T, C, H, W = x.shape
            ws, st = self.window_size, self.stride

            feats = []
            for start in range(0, T, st):
                end = min(start + ws, T)
                win = x[:, start:end]               # (B, win_len, C, H, W)
                win_len = end - start

                # flatten temporal into batch
                win = win.contiguous().view(-1, C, H, W)   # (B * win_len, C, H, W)
                f = self.conv2d(win)                       # (B * win_len, feat_dim)
                f = f.view(B, win_len, -1).permute(0, 2, 1) # (B, feat_dim, win_len)

                feats.append(f)
                # free temporary
                del win, f

            # stitch windows back along time
            framewise = torch.cat(feats, dim=2)             # (B, feat_dim, total_frames)
            seq_lengths = len_x
        else:
            # already per-frame features: expect (B, feat_dim, T)
            framewise    = x
            seq_lengths  = len_x

        # ── 2) Temporal conv → BiLSTM → classifier → decoding ──
        # print(framewise.shape, seq_lengths)
        conv1d_out   = self.conv1d(framewise, seq_lengths)
        x_feat       = conv1d_out['visual_feat']           # (B, hidden, T′)
        lgt          = conv1d_out['feat_len']
        # print(x_feat.shape, lgt)
        tm_out       = self.temporal_model(x_feat, lgt)
        logits       = self.classifier(tm_out['predictions'])  # (T′, B, num_classes)

        if not self.training:
            pred      = self.decoder.decode(logits,    lgt, batch_first=False)
            conv_pred = self.decoder.decode(
                            conv1d_out['conv_logits'], lgt, batch_first=False
                        )
        else:
            pred = conv_pred = None

        return {
            "feat_len":        lgt,
            "conv_logits":     conv1d_out['conv_logits'],
            "sequence_logits": logits,
            "conv_sents":      conv_pred,
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
