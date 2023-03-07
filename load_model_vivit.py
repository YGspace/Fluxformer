import torch
import torch.nn.functional as F
import pytorch_lightning

import torchvision.models.video
import pytorch_lightning
import torch.optim as optim
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
from pretrained_vit import ViT
import pytorch_lightning as pl
import numpy as np


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim=768, depth=4, heads=3, pool='cls',
                 in_channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4, ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        self.space_transformer =ViT('B_16_imagenet1k', pretrained=True,
            image_size=image_size,
            patches=patch_size,
            num_classes=dim,
            dim=dim,
            num_layers=depth,
            num_heads=dim_head,
            dropout_rate=dropout
            )

        # self.space_transformer = ViT(
        #     image_size=image_size,
        #     patch_size=patch_size,
        #     num_classes=num_classes,
        #     dim=dim * scale_dim,
        #     depth=depth,
        #     heads=dim_head,
        #     mlp_dim=dim * scale_dim,
        #     dropout=dropout,
        #     emb_dropout=emb_dropout
        # )



        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))

        self.temporal_transformer = ViT('B_16_imagenet1k', pretrained=True,
            image_size=image_size,
            patches=patch_size,
            num_classes=dim,
            dim=dim,
            num_layers=depth,
            num_heads=dim_head,
            dropout_rate=dropout
            )
        # self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)
        # self.temporal_transformer = ViT(
        #     image_size=image_size,
        #     patch_size=patch_size,
        #     num_classes=num_classes,
        #     dim=dim * scale_dim,
        #     depth=depth,
        #     heads=dim_head,
        #     mlp_dim=dim * scale_dim,
        #     dropout=dropout,
        #     emb_dropout=emb_dropout
        # )

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        #torch.Size([8, 3, 16, 224, 224])
        x = rearrange(x, 'b c t w h -> b t c w h')

        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)


        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)



        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)


class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self,num_class,data_name,lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViViT(224, 16, num_class, 16)
        self.data_name = data_name
    def forward(self, x):
        return self.model(x["video"])

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs = batch["video"]
        labels = batch["label"]
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
