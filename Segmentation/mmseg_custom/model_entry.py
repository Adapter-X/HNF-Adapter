# 快速选择模型
from timm.models import create_model


def select_model(args, config):
    type2model = {
        'ViT-S/16': create_model('vit_small_patch16_224',
                                 checkpoint_path='./pre_weight/ViT-S_16.npz',
                                 num_classes=config['class_num'], drop_path_rate=0.1),
        'ViT-B/16': create_model('vit_base_patch16_224_in21k',
                                 checkpoint_path='./pre_weight/ViT-B_16.npz',
                                 num_classes=config['class_num'], drop_path_rate=0.1),
        'Swin-B': create_model('vit_base_patch16_224_in21k',
                               checkpoint_path='./pre_weight/ViT-B_16.npz',
                               num_classes=config['class_num'], drop_path_rate=0.1),
        'Convnext': create_model("convnext_base_in22k",
                                 checkpoint_path='./pre_weight/convnext_base_22k_224.pth',
                                 num_classes=config['class_num'],
                                 drop_path_rate=0.1)

    }
    model = type2model[args.model_type]
    return model
