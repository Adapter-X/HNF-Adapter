import timm
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer
from timm.models import VisionTransformer

@register_model # 注册模型
def vit_base_patch16_512(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Base (Vit-B/16)
    """
    # 在model_args中对需要部分参数进行修改，此处调整了img_size， patch_size和in_chans
    model_args = dict(img_size=512, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    # vit_tiny_patch16_224是想要加载的预训练权重对应的模型
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model