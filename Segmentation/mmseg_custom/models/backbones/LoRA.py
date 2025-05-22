import torch
from torch import nn
import timm
from mmseg.models.builder import BACKBONES
from timm.models import create_model
from torchinfo import summary
import math

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LoRA(nn.Module):
    def __init__(self, embed_dim, rank=4):
        super().__init__()
        self.rank = rank
        self.lora_down = nn.Linear(embed_dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.lora_down(x))

class AttentionWithLoRA(nn.Module):
    def __init__(self, original_attention, embed_dim, rank=4):
        super().__init__()
        self.original_attention = original_attention
        self.lora_q = LoRA(embed_dim, rank)
        self.lora_k = LoRA(embed_dim, rank)
        self.lora_v = LoRA(embed_dim, rank)
        self.s = 1  # Scaling factor for LoRA output

    def forward(self, x):
        q = self.original_attention.q(x) + self.lora_q(x) * self.s
        k = self.original_attention.k(x) + self.lora_k(x) * self.s
        v = self.original_attention.v(x) + self.lora_v(x) * self.s
        attn_output = self.original_attention.attn(q, k, v)
        return attn_output

def forward_block_with_lora(self, x):
    x = x + self.drop_path(self.attn(self.norm1(x)))
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x

def set_lora(model, rank=4, s=1):
    for layer in model.children():
        if isinstance(layer, timm.models.vision_transformer.Block):
            embed_dim = layer.attn.qkv.weight.shape[1]
            layer.attn = AttentionWithLoRA(layer.attn, embed_dim, rank)
            layer.s = s
            bound_method = forward_block_with_lora.__get__(layer, layer.__class__)
            setattr(layer, 'forward', bound_method)

class LoRA_vit(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.blocks = nn.ModuleList(list(vit.blocks.children()))
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.patch_embed = vit.patch_embed
        self._pos_embed = vit._pos_embed

    def forward(self, x):
        outputs = []
        x_t = self.patch_embed(x)
        x_t = self._pos_embed(x_t)
        for i, block in enumerate(self.blocks):
            x_t = block(x_t)
            if i in {2, 5, 8, 11}:
                B, _, C = x_t.shape
                outputs.append(x_t[:, 1:].reshape(B, 14, 14, C).permute(0, 3, 1, 2).contiguous())
        return outputs

@BACKBONES.register_module()
class LoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_model = create_model('vit_base_patch16_224', checkpoint_path='/media/dl_shouan/ZHITAI/Adapter_test/mmseg_custom/pre_weight/ViT-B_16.npz', drop_path_rate=0.1)
        set_lora(self.vit_model, rank=4)
        self.model = LoRA_vit(self.vit_model)

        # Freeze all parameters except for LoRA and head
        for param in self.vit_model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters
        for layer in self.vit_model.blocks:
            if isinstance(layer.attn, AttentionWithLoRA):
                for param in layer.attn.lora_q.parameters():
                    param.requires_grad = True
                for param in layer.attn.lora_k.parameters():
                    param.requires_grad = True
                for param in layer.attn.lora_v.parameters():
                    param.requires_grad = True

        # Unfreeze the head (classification layer)
        # Assuming vit_model has a head attribute, if not, adjust accordingly
        for param in self.vit_model.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = LoRA()
    device = torch.device('cuda:1')
    model.to(device)
    model_dict = model.state_dict()
    # print(model_dict)
    # for name, module in model.named_modules():
    #     print(name)
    #
    # # for key, weight in model_dict.items():
    #     # if key == 'blocks.4.8.mlp.fc2.weight':
    #     #     print(weight)
    #     # print(key)
    # trainable = []
    # for n, p in model.named_parameters():
    #     if 'LoRA' in n or 'head' in n:
    #         trainable.append(p)
    #     else:
    #         p.requires_grad = False

    summary(model, input_size=(1, 3, 224, 224), device=device)
