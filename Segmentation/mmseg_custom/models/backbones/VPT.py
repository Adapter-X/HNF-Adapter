import torch
from torch import nn
import timm
from mmseg.models.builder import BACKBONES
from timm.models import create_model
from torchinfo import summary


class PromptToken(nn.Module):
    def __init__(self, embed_dim, num_tokens=5):
        super().__init__()
        self.prompt_tokens = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.prompt_tokens, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        prompt_tokens = self.prompt_tokens.expand(B, -1, -1)
        return torch.cat((prompt_tokens, x), dim=1)


class vpt_vit(nn.Module):
    def __init__(self, vit, num_prompt_tokens=5):
        super().__init__()
        self.blocks = nn.ModuleList(list(vit.blocks.children()))
        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop
        self.patch_embed = vit.patch_embed
        self._pos_embed = vit._pos_embed
        self.prompt_token = PromptToken(vit.embed_dim, num_prompt_tokens)

    def forward(self, x):
        outputs = []
        x_t = self.patch_embed(x)
        x_t = self._pos_embed(x_t)
        x_t = self.prompt_token(x_t)  # Add prompt tokens
        for i, block in enumerate(self.blocks):
            x_t = block(x_t)
            if i in {2, 5, 8, 11}:
                B, _, C = x_t.shape
                outputs.append(x_t[:, 6:].reshape(B, 14, 14, C).permute(0, 3, 1, 2).contiguous())
        return outputs


@BACKBONES.register_module()
class VPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_model = create_model('vit_base_patch16_224',
                                      checkpoint_path='/media/dl_shouan/ZHITAI/Adapter_test/mmseg_custom/pre_weight/ViT-B_16.npz',
                                      drop_path_rate=0.1)
        self.model = vpt_vit(self.vit_model)

        # Freeze all parameters except for prompt tokens
        for param in self.vit_model.parameters():
            param.requires_grad = False

        for param in self.model.prompt_token.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = VPT()
    device = torch.device('cuda:1')
    model.to(device)
    summary(model, input_size=(1, 3, 224, 224), device=device)
