# 加载模型与当前模型参数不完全匹配

import torch
from timm.models import create_model

def load_match_dict(model, model_path):
    """加载模型与当前模型参数不完全匹配"""
    pretrain_dict = torch.load(model_path)
    model_dict = model.state_dict()
    pretrain_dict = {k.replace('.module', ''): v for k, v in pretrain_dict.items()}
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                     k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)



def dict2new(model_dict):
        replace_dict = {'layer1.0': 'layer1_0',
                        'layer1.1': 'layer1_1',
                        'layer1.2': 'layer1_2',
                        'layer2.0': 'layer2_0',
                        'layer2.1': 'layer2_1',
                        'layer2.2': 'layer2_2',
                        'layer2.3': 'layer2_3',
                        'layer3.0': 'layer3_0',
                        'layer3.1': 'layer3_1',
                        'layer3.2': 'layer3_2',
                        'layer3.3': 'layer3_3',
                        'layer3.4': 'layer3_4',
                        'layer3.5': 'layer3_5',
                        'layer4.0': 'layer4_0',
                        'layer4.1': 'layer4_1',
                        'layer4.2': 'layer4_2',
                        'blocks.0.norm1': 'blocks0_att.norm1',
                        'blocks.0.attn': 'blocks0_att.attn',
                        'blocks.0.norm2': 'blocks0_mlp.norm2',
                        'blocks.0.mlp': 'blocks0_mlp.mlp',

                        'blocks.1.norm1': 'blocks1_att.norm1',
                        'blocks.1.attn': 'blocks1_att.attn',
                        'blocks.1.norm2': 'blocks1_mlp.norm2',
                        'blocks.1.mlp': 'blocks1_mlp.mlp',

                        'blocks.2.norm1': 'blocks2_att.norm1',
                        'blocks.2.attn': 'blocks2_att.attn',
                        'blocks.2.norm2': 'blocks2_mlp.norm2',
                        'blocks.2.mlp': 'blocks2_mlp.mlp',

                        'blocks.3.norm1': 'blocks3_att.norm1',
                        'blocks.3.attn': 'blocks3_att.attn',
                        'blocks.3.norm2': 'blocks3_mlp.norm2',
                        'blocks.3.mlp': 'blocks3_mlp.mlp',

                        'blocks.4.norm1': 'blocks4_att.norm1',
                        'blocks.4.attn': 'blocks4_att.attn',
                        'blocks.4.norm2': 'blocks4_mlp.norm2',
                        'blocks.4.mlp': 'blocks4_mlp.mlp',

                        'blocks.5.norm1': 'blocks5_att.norm1',
                        'blocks.5.attn': 'blocks5_att.attn',
                        'blocks.5.norm2': 'blocks5_mlp.norm2',
                        'blocks.5.mlp': 'blocks5_mlp.mlp',

                        'blocks.6.norm1': 'blocks6_att.norm1',
                        'blocks.6.attn': 'blocks6_att.attn',
                        'blocks.6.norm2': 'blocks6_mlp.norm2',
                        'blocks.6.mlp': 'blocks6_mlp.mlp',

                        'blocks.7.norm1': 'blocks7_att.norm1',
                        'blocks.7.attn': 'blocks7_att.attn',
                        'blocks.7.norm2': 'blocks7_mlp.norm2',
                        'blocks.7.mlp': 'blocks7_mlp.mlp',

                        'blocks.8': 'blocks8.0',
                        'blocks.9': 'blocks8.1',
                        'blocks.10': 'blocks8.2',
                        'blocks.11': 'blocks8.3',
                        }
        model_dict_new = {}
        for key, value in model_dict.items():
            new_key = key
            for old, new in replace_dict.items():
                new_key = new_key.replace(old, new)
            model_dict_new[new_key] = value

        model_dict.clear()
        model_dict.update(model_dict_new)


def load_dict(model,class_num=100):
    vit_model = create_model('vit_base_patch16_224_in21k', checkpoint_path='./pre_weight/ViT-B_16.npz', drop_path_rate= 0.1)
    vit_model.reset_classifier(class_num)
    torch.save(vit_model.state_dict(), './load_weight/vit-b_16.pth')
    vit_dict = torch.load('./load_weight/vit-b_16.pth')
    dict2new(vit_dict)

    resnet50 = create_model('resnet50', num_classes=11221)
    # resnet50 = create_model('resnet50')
    # resnet50_w = torch.load('./pre_weight/deeplabv3_resnet50.pth')
    # resnet50.load_state_dict(resnet50_w, strict=False)
    resnet50 =load_model_weights(resnet50, './pre_weight/resnet50_21k.pth')
    resnet50.reset_classifier(class_num)
    torch.save(resnet50.state_dict(), './load_weight/resnet50_21k_train.pth')
    resnet50_dict = torch.load('./load_weight/resnet50_21k_train.pth')
    dict2new(resnet50_dict)

    model.load_state_dict(vit_dict, strict=False)
    model.load_state_dict(resnet50_dict, strict=False)

    # vit_model = create_model('vit_small_patch16_224_in21k', checkpoint_path='./pre_weight/ViT-S_16.npz', drop_path_rate=0.1)
    # vit_model.reset_classifier(class_num)
    # torch.save(vit_model.state_dict(), './load_weight/vit-s_16.pth')
    # vit_dict = torch.load('./load_weight/vit-s_16.pth')
    # dict2new(vit_dict)
    #
    # resnet34 = create_model('resnet34', checkpoint_path='./pre_weight/resnet34.pth', drop_path_rate=0.1)
    # resnet34.reset_classifier(class_num)
    # torch.save(resnet34.state_dict(), './load_weight/resnet34_train.pth')
    # resnet34_dict = torch.load('./load_weight/resnet34_train.pth')
    # dict2new(resnet34_dict)

    # model.load_state_dict(vit_dict, strict=False)
    # model.load_state_dict(resnet50_dict, strict=False)

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