import torch
from torch import nn
import timm
import math
from mmseg.models.builder import BACKBONES
from timm.models import create_model
from torchinfo import summary

def forward_block(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.adapter_attn(self.norm1(x))) * self.s
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) * self.s
    return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Adapter(nn.Module):
    def __init__(self, dim=8, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, 768)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up

def set_adapter(model, dim=8, s=1, xavier_init=False):
    for _ in model.children():
        if type(_) == timm.models.vision_transformer.Block:
            _.adapter_attn = Adapter(dim, xavier_init)
            _.adapter_mlp = Adapter(dim, xavier_init)
            _.s = s
            bound_method = forward_block.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)


class Adapter_vit(nn.Module):
    def __init__(self, vit):
        # Transformer
        super().__init__()
        self.blocks = nn.ModuleList(list(vit.blocks.children()))

        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop

        self.patch_embed = vit.patch_embed
        self._pos_embed = vit._pos_embed

    def forward(self, x):
        x_t = self.patch_embed(x)
        x_t = self._pos_embed(x_t)
        x_t = self.blocks[0](x_t)
        x_t = self.blocks[1](x_t)
        x_t = self.blocks[2](x_t)
        B, _, C = x_t.shape
        x_t_1 = x_t[:, 1:].reshape(B, 14, 14,
                          768).permute(0, 3, 1, 2).contiguous()

        x_t = self.blocks[3](x_t)
        x_t = self.blocks[4](x_t)
        x_t = self.blocks[5](x_t)
        x_t_2 = x_t[:, 1:].reshape(B, 14, 14,
                          768).permute(0, 3, 1, 2).contiguous()

        x_t = self.blocks[6](x_t)
        x_t = self.blocks[7](x_t)
        x_t = self.blocks[8](x_t)
        x_t_3 = x_t[:, 1:].reshape(B, 14, 14,
                          768).permute(0, 3, 1, 2).contiguous()

        x_t = self.blocks[9](x_t)
        x_t = self.blocks[10](x_t)
        x_t = self.blocks[11](x_t)
        x_t_4 = x_t[:, 1:].reshape(B, 14, 14,
                          768).permute(0, 3, 1, 2).contiguous()

        return [x_t_1, x_t_2, x_t_3, x_t_4]

@BACKBONES.register_module()
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_model = create_model('vit_base_patch16_224',
                                      checkpoint_path='/media/dl_shouan/ZHITAI/Adapter_test/mmseg_custom/pre_weight/ViT-B_16.npz',
                                      drop_path_rate=0.1)
        set_adapter(self.vit_model, dim=8)

        self.model = Adapter_vit(self.vit_model)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = Adapter()
    device = torch.device('cuda:1')
    model.to(device)
    summary(model, input_size=(1, 3, 224, 224), device=device)