import timm.models.vision_transformer
import torch
import torch.nn as nn
from functools import partial
from timm.models import create_model
from timm.models.layers import DropBlock2d
import types
from mmseg.models.builder import BACKBONES
from torchinfo import summary
import math
from .base.vit import vit_base_patch16_512

def find_nearest(D, S):
    if S > D:
        while abs(S - D) > abs(S // 2 - D):
            S //= 2
    else:
        while abs(S - D) > abs(S * 2 - D):
            S *= 2
    return S


def resize_conv2d(input_tensor, target_size, xavier=False):
    """
    使用卷积下采样实现resize操作

    参数：
    - input_tensor: 输入图像的张量，形状为 (batch_size, channels, height, width)
    - target_size: 目标图像的大小，形状为 (target_height, target_width)

    返回：
    - 调整大小后的图像张量
    """
    global conv_layer
    target_height, target_width = target_size
    # 创建卷积层，将步幅设置为适当的值，以实现下采样
    if input_tensor.shape[2] > target_height:
        # 计算整数步幅
        stride_height = input_tensor.shape[2] // target_height
        stride_width = input_tensor.shape[3] // target_width
        conv_layer = nn.Conv2d(input_tensor.shape[1], input_tensor.shape[1], kernel_size=2 * stride_width,
                               stride=(stride_height, stride_width), padding=stride_width // 2,
                               groups=input_tensor.shape[1])
    if input_tensor.shape[2] <= target_height:
        # 计算整数步幅
        stride_height = target_height // input_tensor.shape[2]
        stride_width = target_width // input_tensor.shape[3]
        conv_layer = nn.ConvTranspose2d(input_tensor.shape[1], input_tensor.shape[1], kernel_size=stride_height,
                                        stride=(stride_height, stride_width), groups=input_tensor.shape[1])
    if xavier == True:
        nn.init.xavier_uniform_(conv_layer.weight)
    else:
        nn.init.zeros_(conv_layer.weight)
    nn.init.zeros_(conv_layer.bias)
    device = input_tensor.device
    conv_layer = conv_layer.to(device)
    # 执行卷积操作
    resized_image = conv_layer(input_tensor)

    return resized_image


class Convchor(nn.Module):
    def __init__(self, ch_in, f, ch_out, stride, r, padding=1, act_layer=nn.ReLU, norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(Convchor, self).__init__()
        self.pointwise_conv1 = nn.Conv2d(ch_in, f, kernel_size=1, stride=1)
        nn.init.xavier_uniform_(self.pointwise_conv1.weight)
        nn.init.zeros_(self.pointwise_conv1.bias)
        self.bn1 = norm_layer(f)
        self.act1 = act_layer(inplace=True)

        self.depthwise_conv = nn.Conv2d(f, f, kernel_size=3, stride=stride, padding=padding, groups=f)
        nn.init.xavier_uniform_(self.depthwise_conv.weight)
        nn.init.zeros_(self.depthwise_conv.bias)
        self.bn2 = norm_layer(f)
        self.act2 = act_layer(inplace=True)

        self.pointwise_conv2 = nn.Conv2d(f, ch_out, kernel_size=1, stride=1)
        nn.init.xavier_uniform_(self.pointwise_conv2.weight)
        nn.init.zeros_(self.pointwise_conv2.bias)

        self.bn3 = norm_layer(ch_out)
        self.act3 = act_layer(inplace=True)
        self.r = r

    def forward(self, x):
        x = self.pointwise_conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.pointwise_conv2(x)
        x = self.bn3(x)
        x = self.act3(x)

        x = self.r * x
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def drop_blocks(drop_block_rate=0.):
    return DropBlock2d(drop_block_rate, 5, 0.25) if drop_block_rate else None


class Transchor(nn.Module):
    def __init__(self, embed_dim, h=8, cat=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.depthwise_conv = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1,
                                        groups=self.embed_dim)

        nn.init.xavier_uniform_(self.depthwise_conv.weight)
        nn.init.zeros_(self.depthwise_conv.bias)

        self.bn = nn.BatchNorm2d(self.embed_dim)

        if cat == False:
            self.adapter_down = nn.Linear(self.embed_dim, h)  # equivalent to 1 * 1 Conv
        else:
            self.adapter_down = nn.Linear(self.embed_dim * 2, h)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(h, self.embed_dim)  # equivalent to 1 * 1 Conv
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.xavier_uniform_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.h = h

    def forward(self, x):
        B, N, C = x.shape

        x_patch = x[:, 1:].reshape(B, int(math.sqrt(N-1)), int(math.sqrt(N-1)), self.embed_dim).permute(0, 3, 1, 2)
        x_cls = x[:, :1]

        x_patch = self.depthwise_conv(x_patch)
        x_patch = self.bn(x_patch)
        x_patch = self.act(x_patch)

        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, N-1, self.embed_dim)

        x = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        return x_up


class Bridge_ct(nn.Module):
    def __init__(self, N, outplanes, embed_dim, m=8, a=1, act_layer=nn.GELU):
        super().__init__()
        self.embed_dim = embed_dim
        self.repatch_H = N

        # 降维
        self.adapter_down = nn.Conv2d(outplanes, m, 1, 1, 0)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        self.bn1 = nn.BatchNorm2d(m)
        self.act1 = act_layer()

        # 升维
        self.adapter_up = nn.Conv2d(m, 768, 1, 1, 0)
        nn.init.xavier_uniform_(self.adapter_up.weight)
        self.bn2 = nn.BatchNorm2d(768)
        self.act2 = act_layer()
        self.drop = nn.Dropout(0.1)
        self.a = a

    def forward(self, x_r):
        B, C, H, W = x_r.shape

        x_r = self.adapter_down(x_r)
        x_r = self.bn1(x_r)
        x_r = self.act1(x_r)

        x_r = resize_conv2d(x_r, (self.repatch_H, self.repatch_H), xavier=True)

        x_r = self.adapter_up(x_r)
        x_r = self.bn2(x_r)
        x_r = self.act2(x_r)

        x_r = x_r.permute(0, 2, 3, 1).reshape(B, self.repatch_H * self.repatch_H, self.embed_dim)
        x_cls = x_r[:, :1]
        x_t = torch.cat([x_cls, x_r], dim=1)
        x_t = self.a * x_t
        return x_t


class Bridge_tc(nn.Module):
    def __init__(self, outplanes, embed_dim, H, l=8, b=1, act_layer=nn.GELU):
        super().__init__()
        self.H = H
        self.embed_dim = embed_dim

        # 降维
        self.adapter_down = nn.Conv2d(768, l, 1, 1, 0)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        self.bn1 = nn.BatchNorm2d(l)
        self.act1 = act_layer()

        # 升维
        self.adapter_up = nn.Conv2d(l, outplanes, 1, 1, 0)
        nn.init.xavier_uniform_(self.adapter_up.weight)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.act2 = act_layer()
        self.drop = nn.Dropout(0.1)

        self.b = b

    def forward(self, x_t):
        B, N, D = x_t.shape
        x_patch = x_t[:, 1:].reshape(B, int(math.sqrt(N-1)), int(math.sqrt(N-1)), self.embed_dim).permute(0, 3, 1, 2)

        x_t = self.adapter_down(x_patch)
        x_t = self.bn1(x_t)
        x_t = self.act1(x_t)

        x_r = resize_conv2d(x_t, (self.H, self.H), xavier=True)

        x_r = self.adapter_up(x_r)
        x_r = self.bn2(x_r)
        x_r = self.act2(x_r)

        x_t = self.b * x_r
        return x_t


def forward_conv(self, x):
    shortcut = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)

    x1 = self.conv_adapter(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.drop_block(x)
    x = self.act2(x)
    x = self.aa(x)

    x = x + x1

    x = self.conv3(x)
    x = self.bn3(x)

    if self.se is not None:
        x = self.se(x)

    if self.drop_path is not None:
        x = self.drop_path(x)

    if self.downsample is not None:
        shortcut = self.downsample(shortcut)
    x += shortcut
    x = self.act3(x)

    return x



# def forward_conv(self, x):
#     shortcut = x
#
#     x1 = self.conv_adapter(x)
#     x = self.conv1(x)
#     x = self.bn1(x)
#     if self.drop_block is not None:
#         x = self.drop_block(x)
#     x = self.act1(x)
#     if self.aa is not None:
#         x = self.aa(x)
#
#     x = self.conv2(x)
#     x = self.bn2(x)
#     if self.drop_block is not None:
#         x = self.drop_block(x)
#
#     if self.drop_path is not None:
#         x = self.drop_path(x)
#
#     if self.downsample is not None:
#         shortcut = self.downsample(shortcut)
#     x = x + x1
#     x += shortcut
#     x = self.act2(x)
#
#     return x


def forward_mlp(self, x):
    x1 = self.adapter_mlp(x)
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    x = x + x1
    return x


def forward_atten(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x1 = self.adapter_attn(x)
    x = self.proj(x)
    x = x + x1
    x = self.proj_drop(x)
    return x


def set_adapter(model, adapter, embed_dim, f=8, h=8, r=1, s=0.1):
    if adapter == "Convchor":
        for _ in model.modules():
            if type(_) == timm.models.resnet.Bottleneck:
                ch_in = _.conv2.in_channels
                ch_out = _.conv2.out_channels
                stride = _.conv2.stride
                _.conv_adapter = Convchor(ch_in=ch_in, f=f, ch_out=ch_out, stride=stride, r=r)
                bound_method = forward_conv.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)

    if adapter == "Transchor":
        for _ in model.modules():
            if type(_) == timm.models.vision_transformer.Attention:
                _.adapter_attn = Transchor(embed_dim=embed_dim, h=h)
                _.s = s
                bound_method = forward_atten.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            if type(_) == timm.models.vision_transformer.Mlp:
                _.adapter_mlp = Transchor(embed_dim=embed_dim, h=h)
                _.s = s
                bound_method = forward_mlp.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)


class Base_50_Bridge(nn.Module):
    def __init__(self, cnn, vit, m=8, l=8, a=1, b=1, embed_dim=768, N_H=14):
        # Transformer
        super().__init__()

        self.conv1 = cnn.conv1
        self.bn1 = cnn.bn1
        self.act1 = cnn.act1
        self.maxpool = cnn.maxpool

        self.cls_token = vit.cls_token
        self.pos_embed = vit.pos_embed
        self.pos_drop = vit.pos_drop

        self.patch_embed = vit.patch_embed
        self._pos_embed = vit._pos_embed

        # 获取 ResNet50 的 4 个阶段
        self.layer1 = cnn.layer1
        self.layer2 = cnn.layer2
        self.layer3 = cnn.layer3
        self.layer4 = cnn.layer4

        # 获取 ViT-b/16 的 12 个块
        self.blocks = nn.ModuleList(list(vit.blocks.children()))

        self.adapter_ct1 = Bridge_ct(N_H, 256, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct2 = Bridge_ct(N_H, 512, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct3 = Bridge_ct(N_H, 512, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct4 = Bridge_ct(N_H, 512, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct5 = Bridge_ct(N_H, 512, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct6 = Bridge_ct(N_H, 1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct7 = Bridge_ct(N_H, 1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct8 = Bridge_ct(N_H, 1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct9 = Bridge_ct(N_H, 1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct10 = Bridge_ct(N_H, 1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct11 = Bridge_ct(N_H, 1024, embed_dim=embed_dim, m=m, a=a)
        self.adapter_ct12 = Bridge_ct(N_H, 2048, embed_dim=embed_dim, m=m, a=a)
        if N_H==14:
            self.adapter_tc1 = Bridge_tc(256, embed_dim=embed_dim, H=56, l=l, b=b)
            self.adapter_tc2 = Bridge_tc(512, embed_dim=embed_dim, H=28, l=l, b=b)
            self.adapter_tc3 = Bridge_tc(512, embed_dim=embed_dim, H=28, l=l, b=b)
            self.adapter_tc4 = Bridge_tc(512, embed_dim=embed_dim, H=28, l=l, b=b)
            self.adapter_tc5 = Bridge_tc(512, embed_dim=embed_dim, H=28, l=l, b=b)
            self.adapter_tc6 = Bridge_tc(1024, embed_dim=embed_dim, H=14, l=l, b=b)
            self.adapter_tc7 = Bridge_tc(1024, embed_dim=embed_dim, H=14, l=l, b=b)
            self.adapter_tc8 = Bridge_tc(1024, embed_dim=embed_dim, H=14, l=l, b=b)
            self.adapter_tc9 = Bridge_tc(1024, embed_dim=embed_dim, H=14, l=l, b=b)
            self.adapter_tc10 = Bridge_tc(1024, embed_dim=embed_dim, H=14, l=l, b=b)
            self.adapter_tc11 = Bridge_tc(1024, embed_dim=embed_dim, H=14, l=l, b=b)
            self.adapter_tc12 = Bridge_tc(2048, embed_dim=embed_dim, H=7, l=l, b=b)
        else:
            self.adapter_tc1 = Bridge_tc(256, embed_dim=embed_dim, H=128, l=l, b=b)
            self.adapter_tc2 = Bridge_tc(512, embed_dim=embed_dim, H=64, l=l, b=b)
            self.adapter_tc3 = Bridge_tc(512, embed_dim=embed_dim, H=64, l=l, b=b)
            self.adapter_tc4 = Bridge_tc(512, embed_dim=embed_dim, H=64, l=l, b=b)
            self.adapter_tc5 = Bridge_tc(512, embed_dim=embed_dim, H=64, l=l, b=b)
            self.adapter_tc6 = Bridge_tc(1024, embed_dim=embed_dim, H=32, l=l, b=b)
            self.adapter_tc7 = Bridge_tc(1024, embed_dim=embed_dim, H=32, l=l, b=b)
            self.adapter_tc8 = Bridge_tc(1024, embed_dim=embed_dim, H=32, l=l, b=b)
            self.adapter_tc9 = Bridge_tc(1024, embed_dim=embed_dim, H=32, l=l, b=b)
            self.adapter_tc10 = Bridge_tc(1024, embed_dim=embed_dim, H=32, l=l, b=b)
            self.adapter_tc11 = Bridge_tc(1024, embed_dim=embed_dim, H=32, l=l, b=b)
            self.adapter_tc12 = Bridge_tc(2048, embed_dim=embed_dim, H=16, l=l, b=b)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_r = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        x_t = self.patch_embed(x)
        x_t = self._pos_embed(x_t)

        # 1
        x_r = self.layer1[0](x_r)
        x_r = self.layer1[1](x_r)
        x_ct = self.adapter_ct1(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[0](x_t)

        x_tc = self.adapter_tc1(x_t)
        x_r = x_r + x_tc

        # 2
        x_r = self.layer1[2](x_r)
        x_r = self.layer2[0](x_r)
        x_ct = self.adapter_ct2(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[1](x_t)

        x_tc = self.adapter_tc2(x_t)
        x_r1 = x_r + x_tc

        # 3
        x_r = self.layer2[1](x_r1)
        x_ct = self.adapter_ct3(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[2](x_t)

        x_tc = self.adapter_tc3(x_t)
        x_r = x_r + x_tc

        # 4
        x_r = self.layer2[2](x_r)
        x_ct = self.adapter_ct4(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[3](x_t)

        x_tc = self.adapter_tc4(x_t)
        x_r = x_r + x_tc

        # 5
        x_r = self.layer2[3](x_r)
        x_ct = self.adapter_ct5(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[4](x_t)

        x_tc = self.adapter_tc5(x_t)
        x_r2 = x_r + x_tc

        # 6
        x_r = self.layer3[0](x_r2)
        x_ct = self.adapter_ct6(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[5](x_t)

        x_tc = self.adapter_tc6(x_t)
        x_r = x_r + x_tc

        # 7
        x_r = self.layer3[1](x_r)
        x_ct = self.adapter_ct7(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[6](x_t)

        x_tc = self.adapter_tc7(x_t)
        x_r = x_r + x_tc

        # 8
        x_r = self.layer3[2](x_r)
        x_ct = self.adapter_ct8(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[7](x_t)

        x_tc = self.adapter_tc8(x_t)
        x_r = x_r + x_tc

        # 9
        x_r = self.layer3[3](x_r)
        x_ct = self.adapter_ct9(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[8](x_t)

        x_tc = self.adapter_tc9(x_t)
        x_r = x_r + x_tc

        # 10
        x_r = self.layer3[4](x_r)
        x_ct = self.adapter_ct10(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[9](x_t)

        x_tc = self.adapter_tc10(x_t)
        x_r = x_r + x_tc

        # 11
        x_r = self.layer3[5](x_r)
        x_ct = self.adapter_ct11(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[0](x_t)

        x_tc = self.adapter_tc11(x_t)
        x_r3 = x_r + x_tc

        # 12
        x_r = self.layer4[0](x_r3)
        x_r = self.layer4[1](x_r)
        x_r = self.layer4[2](x_r)
        x_ct = self.adapter_ct12(x_r)
        x_t = x_t + x_ct
        x_t = self.blocks[11](x_t)

        x_tc = self.adapter_tc12(x_t)
        x_r4 = x_r + x_tc

        return [x_r1, x_r2, x_r3, x_r4]


def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
    return model

@BACKBONES.register_module()
class Resnet50_ViT_B(nn.Module):
    """
    Resnet50和ViT-B/16

    参数：
    - f: Convhor隐藏层维数
    - h: Tranchor隐藏层维数
    - m: Bridge_ct隐藏层维数
    - l: Bridge_tc隐藏层维数
    - r: Conchor输出系数
    - s: Tranchor输出系数
    - a: Bridge_ct输出系数
    - b: Bridge_tc输出系数
    """

    def __init__(self, f, h, m, l, r, s, a, b, img_size=224):
        super().__init__()
        self.resnet50 = create_model('resnet50', num_classes=11221, drop_path_rate=0.1)
        self.resnet50 = load_model_weights(self.resnet50, '.../mmseg_custom/pre_weight/resnet50_21k.pth')
        if img_size == 512:
            self.vit_model = create_model('vit_base_patch16_512',
                                      checkpoint_path='.../mmseg_custom/pre_weight/ViT-B_16.npz',
                                      drop_path_rate=0.1)
        else:
            self.vit_model = create_model('vit_base_patch16_224',
                                      checkpoint_path='.../mmseg_custom/pre_weight/ViT-B_16.npz',
                                      drop_path_rate=0.1)
        embed_dim = self.vit_model.embed_dim
        set_adapter(self.resnet50, "Convchor", embed_dim=embed_dim, f=f, h=h, r=r, s=s)
        set_adapter(self.vit_model, "Transchor", f=f, h=h, embed_dim=embed_dim, r=r, s=s)
        N_H = img_size//16

        self.model = Base_50_Bridge(cnn=self.resnet50, vit=self.vit_model, m=m, l=l, a=a, b=b,
                                     embed_dim=embed_dim, N_H = N_H)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = Resnet50_ViT_B(f=80, h=8, m=8, l=72, r=0.1, s=0.1, a=0.1, b=1, img_size=224)
    device = torch.device('cuda:1')
    model.to(device)
    model_dict = model.state_dict()

    trainable = []
    for n, p in model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable.append(p)
        else:
            p.requires_grad = False

    summary(model, input_size=(1, 3, 224, 224), device=device)
