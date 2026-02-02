import timm
from timm.layers import resample_abs_pos_embed_nhwc
import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        y = self.weight[:, None, None] * x
        # y = torch.mul(self.weight[:, None, None], x)
        x = y + self.bias[:, None, None]
        return x


class ViT(nn.Module):
    def __init__(self, 
    encoder_embed_dim: int = 768,
    pretrain_model: str = 'samvit_base_patch16',
    out_chans: int = 256,
    depth: int = 12,
    pretrained: bool = True,
    freeze_encoder: bool = True,
    ) -> None:

        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.depth = depth
        self.pretrain_model = pretrain_model
        self.sam_encoder = timm.create_model(self.pretrain_model, pretrained=pretrained, num_classes=0)
        
        if freeze_encoder:
            for name, param in self.sam_encoder.named_parameters():
                param.requires_grad = False

        self.neck = nn.Sequential(
            nn.Conv2d(self.encoder_embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sam_encoder.patch_embed(x)
        if self.sam_encoder.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.sam_encoder.pos_embed, x.shape[1:3])
        x = self.sam_encoder.pos_drop(x)
        x = self.sam_encoder.patch_drop(x)
        x = self.sam_encoder.norm_pre(x)

        for i in range(self.depth):
            x = self.sam_encoder.blocks[i](x)
        x = self.neck(x.permute(0, 3, 1, 2))

        return x

if __name__ == '__main__':
    x = torch.rand((4,3,256,256))
    model = ViT()
    print(model(x).shape)