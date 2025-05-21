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
from transformers import T5ForConditionalGeneration, T5TokenizerFast

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

class NTXent(nn.Module):
    """Cosine-similarity InfoNCE loss (SimCLR)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.T = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        # positives: diagonal of cosine-sim matrix
        pos = (z1 * z2).sum(dim=1) / self.T                # [B]

        # full similarity matrix: [2B, 2B]
        z = torch.cat([z1, z2], dim=0)                     # [2B, D]
        sim = (z.float() @ z.float().T) / self.T
        sim.fill_diagonal_(float('-inf'))                           # mask self-sim

        # InfoNCE
        log_prob = pos.float() - sim.logsumexp(dim=1)[: z1.size(0)]
        return -log_prob.mean().to(pos.dtype)

# decoder  = ctc_decoder(lexicon = None,tokens=vocab, blank_token=tokens["id2token"][1], sil_token=tokens["id2token"][1], unk_word=tokens['id2token'][0], beam_size=10, lm=None)  # add lm_path=... for KenLM
# beam_out = decoder(logits.cpu().float().contiguous(), input_lens.cpu().float().contiguous())
# best_hyps = [h[0].tokens for h in beam_out]                  # beam_out is a list per sample


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
        t5_name="t5-base",              # 768-dim hidden
        freeze_t5=True,                 # start frozen – unfreeze gradually
        contrast_dim=256,
        contrast_tau=0.07,
        contrast_lambda=0.1,        # weight of contrastive branch
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weights = loss_weights or {}
        self.window_size = window_size
        self.stride = stride

        # CNN frontend (3D ResNet)
        self.conv2d = getattr(resnet, c2d_type)()
        # self.conv2d.requires_grad_(False)
        self.conv2d.fc = Identity()

        # 1D convolution temporal extractor
        self.conv1d = TemporalConv(
            input_size=512 if c2d_type in ['resnet18', 'resnet34'] else 2048,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes,
        )

        self.c2d_proj = nn.Sequential(                 # view 1  (after ResNet)
            nn.AdaptiveAvgPool1d(1),  # T-pool
            nn.Flatten(),             # [B, C]
            nn.Linear(512 if c2d_type in ['resnet18', 'resnet34'] else 2048, contrast_dim),
            nn.SiLU(),
            nn.Linear(contrast_dim, contrast_dim)
        )
        self.c1d_proj = nn.Sequential(                 # view 2 (after TemporalConv)
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, contrast_dim),
            nn.SiLU(),
            nn.Linear(contrast_dim, contrast_dim)
        )
        self.contrast_loss = NTXent(contrast_tau)
        self.contrast_lambda = contrast_lambda

        # self.temporal_model = BiLSTMLayer(
        #     # rnn_type="LSTM",
        #     input_size=hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=2,
        #     bidirectional=True,
        #     dropout=0,
        # )

        self.proj = nn.Linear(hidden_size, 768)        # T5-base hidden size
        self.act   = nn.SiLU()
        self.norm  = nn.LayerNorm(768)
        self.t5   = T5ForConditionalGeneration.from_pretrained(t5_name)
        # if freeze_t5:
        #     self.t5.requires_grad_(False)              # later unfreeze last k blocks

        # tokenizer is kept outside of the module so DDP can share one copy
        self.tokenizer = T5TokenizerFast.from_pretrained(t5_name)

        self.t5.requires_grad_(False)          # whole model frozen

        n = 0
        enc_blocks = self.t5.encoder.block
        n = min(n, len(enc_blocks))

        if n:
            for blk in enc_blocks[:n]:
                for p in blk.parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in self.t5.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.t5.parameters())

        print(f"[T5-unfreeze]  last {n} encoder layers now trainable "
            f"→ {trainable/1e6:.1f} M / {total/1e6:.1f} M params")

        cls_layer = NormLinear if weight_norm else nn.Linear
        self.classifier = cls_layer(hidden_size, num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        else:
            self.conv1d.fc = cls_layer(hidden_size, num_classes)

        self.decoder = utils.Decode(gloss_dict, num_classes, search_mode="beam")
        print(num_classes)

        # self.loss = {
        #     "CTCLoss": nn.CTCLoss(reduction="none", zero_infinity=False),
        #     "distillation": SeqKD(T=8),
        # }

    def forward(self, x, len_x, text=None) -> ModelOutput:
        # print("input", x.shape)
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            #inputs = x.reshape(batch * temp, channel, height, width)
            #framewise = self.masked_bn(inputs, len_x)
            #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        # tm_outputs = self.temporal_model(x, lgt)
        # outputs = self.classifier(tm_outputs['predictions'])
        # pred = None if self.training \
        #     else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        # conv_pred = None if self.training \
        #     else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)


        z1 = self.c2d_proj(framewise)          # framewise: [B, C, T]
        # View 2: TemporalConv features (x : [T, B, C] -> [B, C, T])
        z2 = self.c1d_proj(x.permute(1, 2, 0))

        contr_loss = self.contrast_loss(z1, z2) if self.training else 0.0

        feats_proj = self.norm(self.proj(self.act(x))).permute(1, 0, 2) # (B, T, 768)

        if text is not None:                       # training branch
            tok = self.tokenizer(
                text, padding="longest", return_tensors="pt"
            ).to(feats_proj.device)

            out = self.t5(
                inputs_embeds       = feats_proj,
                decoder_input_ids   = tok.input_ids,
                labels              = tok.input_ids,
            )
            loss   = out.loss
            preds  = out.logits.argmax(-1)
        else:                                                # inference
            preds = self.t5.generate(inputs_embeds=feats_proj,
                                    decoder_start_token_id=self.tokenizer.pad_token_id,
                                      num_beams=4)
            loss = None
        
        pred_text = self.tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        return {"loss": (loss, contr_loss), "pred_ids": preds, 'pred_text': pred_text,
                # keep these if you still want the old auxiliary heads
                "conv_logits": conv1d_outputs['conv_logits'],
                "feat_len": lgt}

        # return {
        #     #"framewise_features": framewise,
        #     #"visual_features": x,
        #     "feat_len": lgt,
        #     "conv_logits": conv1d_outputs['conv_logits'],
        #     "sequence_logits": outputs,
        #     "conv_sents": conv_pred,
        #     "recognized_sents": pred,
        # }

    def criterion_calculation(self, output: ModelOutput, label, label_lgt):
        label      = label.to(dtype=torch.long)
        label_lgt  = label_lgt.to(dtype=torch.long)

        feat_len   = output["feat_len"]
        if not torch.is_tensor(feat_len):
            feat_len = torch.tensor(feat_len, device=label.device)
        feat_len   = torch.round(feat_len).to(dtype=torch.long)

        total_loss = 0
        for name, weight in self.loss_weights.items():
            if name == "ConvCTC":
                log_probs = output["conv_logits"].log_softmax(-1)
                # print("ConvCTC loss", log_probs.shape, label.shape, feat_len.shape, label_lgt.shape)
                loss = self.loss["CTCLoss"](log_probs, label, feat_len, label_lgt).mean()

            elif name == "SeqCTC":
                log_probs = output["sequence_logits"].log_softmax(-1)
                # print("SeqCTC loss", log_probs.shape, label.shape, feat_len.shape, label_lgt.shape)
                loss = self.loss["CTCLoss"](log_probs, label, feat_len, label_lgt).mean()

            elif name == "Dist":
                # print("Dist loss", output["conv_logits"].shape, output["sequence_logits"].shape)
                loss = self.loss["distillation"](
                    output["conv_logits"],
                    output["sequence_logits"].detach(),
                    use_blank=False,
                )
            else:
                continue

            total_loss += weight * loss

        return total_loss
