import torch.optim
import torch.nn.functional as F
import torch.utils.data
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm
from loss import CharbonnierLoss
from avalanche.evaluation.metrics.accuracy import Accuracy
from utils.torch_utils import load_match_dict, load_dict
from options import prepare_train_args
from utils.classification_utils import *
from vtab import *
from execl import extract_log_files
from model.HNF import Resnet50_ViT_B
from torchinfo import summary

# 每个epoch训练
def train(model, dl, opt, scheduler, epoch):
    # switch to train mode
    model.train()  # 启用BN和dropout
    model = model.to(device)
    for ep in range(epoch):
        model.train()
        model = model.to(device)
        loop = tqdm(dl, desc='Train:')  # 进度条
        running_loss = 0.0
        for i, data in enumerate(loop):
            img, label = data[0].to(device), data[1].to(device)
            outputs = model(img)
            # 计算loss
            loss_list = [compute_loss(o, label) / len(outputs) for o in outputs]
            loss = sum(loss_list)

            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            loop.set_description(f'Epoch [{ep + 1}/{epoch}]')
            loop.set_postfix(loss=running_loss / (i + 1))
        if scheduler is not None:
            scheduler.step(ep)
        if not os.path.exists(args.model_dir + '/weights'):
            os.system('mkdir -p ' + args.model_dir + '/weights')
        if ep % 5 == 4:
            acc = val(model, val_loader)
            if acc >= config['best_acc']:
                config['best_acc'] = acc
                config['best_epoch'] = ep + 1
                torch.save(model.state_dict(), args.model_dir + '/weights/' + args.data_type + '.pth')
                with open(args.model_dir + '/weights/' + args.data_type + '.log', 'w') as f:
                    f.write(str(ep + 1) + ' ' + str(acc))
            print('best_epoch:', config['best_epoch'], 'best_acc:', config['best_acc'])
    torch.save(model.state_dict(), args.model_dir + '/weights/' + args.data_type + 'latest.pth')


##Evaluation (Validation)
def val(model, dl):
    model.eval()
    model = model.to(device)
    loop = tqdm(dl, desc='Val:')
    acc = Accuracy()
    for i, data in enumerate(loop):
        with torch.no_grad():
            img, label = data[0].to(device), data[1].to(device)
            output = model(img)
            acc.update(output.argmax(dim=1).view(-1), label)
    print('acc:', acc.result())
    return acc.result()


# 计算loss
def compute_loss(pred, gt):
    if args.loss == 'l1':
        loss = CharbonnierLoss()
    elif args.loss == 'ce':
        loss = F.cross_entropy(pred, gt)
    else:
        loss = F.mse_loss(pred, gt).cuda()
    return loss


if __name__ == '__main__':
    dataset_config = ['caltech101',
                      'cifar',
                      'dtd',
                      'oxford_flowers102',
                      'oxford_iiit_pet',
                      'patch_camelyon',
                      'sun397',
                      'svhn',
                      'resisc45',
                      'eurosat',
                      'dmlab',
                      'kitti',
                      'smallnorb_azi',
                      'smallnorb_ele',
                      'dsprites_loc',
                      'dsprites_ori',
                      'clevr_count',
                      'clevr_dist',
                      'diabetic_retinopathy'
                      ]

    for i in dataset_config:
        args = prepare_train_args()  # 初始化命令行参数
        set_seed(args.seed)
        # logger = Logger(args)  # 初始化日志
        gpus = args.gpus
        device = torch.device(gpus)
        args.data_type = i
        config = get_config(args.data_type)
        train_loader, val_loader = get_data(args.data_type, args.batch_size)  # 初始化训练验证的两个dataloader
        print(args.data_type)
        model = Resnet50_ViT_B(f=8, h=8, m=8, l=8, r=0.1, s=0, a=0.1, b=0.1, num_classes=config['class_num'], img_size=224)
        # 加载模型（load_model_path有权重时）
        if args.load_model_path != '':
            print("=> using pre-trained weights for DPSNet")
            if args.load_not_strict:
                load_match_dict(model, args.load_model_path)
            else:
                model.load_state_dict(torch.load(args.load_model_path).state_dict())
        else:
            load_dict(model, config['class_num'])

        config['best_acc'] = 0.0
        trainable = []
        for n, p in model.named_parameters():
            if 'head' in n or 'adapter' in n:
                trainable.append(p)
            else:
                p.requires_grad = False

        optimizer = torch.optim.AdamW(trainable, args.lr, weight_decay=args.weight_decay)  # 优化器
        scheduler = CosineLRScheduler(optimizer, t_initial=100, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
        model = model.to(device)
        summary(model, input_size=(1, 3, 224, 224))
        train(model, train_loader, optimizer, scheduler, args.epochs)
        extract_log_files(args.model_dir + '/weights/', args.model_dir + 'weights.txt')
