import argparse
import os


def parse_common_args(parser):
    parser.add_argument('--folder_type', type=str, default='Classification')  # 文件夹名
    parser.add_argument('--model_type', type=str, default='Conformer', help='used in model_entry.py')  # 模型名
    parser.add_argument('--data_type', type=str, default='vtab', help='used in data_entry.py')  # 数据集名
    parser.add_argument('--save_prefix', type=str, default='B_full',
                        help='some comment for model or test result dir')  # 实验名
    parser.add_argument('--load_model_path', type=str, default='/media/dl_shouan/DATA/fxl/pre_weight/Conformer_base_patch16.pth',
                        help='model path for pretrain or test')  # 模型加载路径
    parser.add_argument('--load_not_strict', action='store_true',
                        help='allow to load only common state dicts')  # 模型加载参数匹配
    parser.add_argument('--val_list', type=str, default='',
                        help='val list in train, test list path in test')  # list
    parser.add_argument('--v', type=str, default='', help='qkv的维度')
    parser.add_argument('--gpus', type=str, default='cuda:1')  # gup选择，单机多卡
    parser.add_argument('--seed', type=int, default=42)  # 随机种子
    parser.add_argument('--dim', type=int, default=8)  # dim
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--miaoshu', type=str, default='Conformer-B在vtab数据集上full')
    return parser


def parse_train_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.95, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--model_dir', type=str, default='', help='leave blank, auto generated')
    parser.add_argument('--train_list', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    return parser


def parse_vit_args(parser):
    parser.add_argument('--vit_patch_size', type=int, default=16,
                        help=" better be a multiple of 2 like 8, 16 etc ..")
    parser.add_argument('--image_size', default=(224, 224))
    parser.add_argument('--vit_model_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--split_size', type=int, default=224,
                             help="better be a multiple of 8, like 128, 256, etc ..")
    return parser


def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')  # 可视化开关
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')  # 结果目录
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    parser = parse_vit_args(parser)
    args = parser.parse_args()
    return args


def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    parser = parse_vit_args(parser)
    args = parser.parse_args()
    return args


def get_train_model_dir(args):
    model_dir = os.path.join('runs', args.model_type, args.data_type, args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir


def get_test_result_dir(args):
    ext = os.path.basename(args.load_model_path).split('.')[-1]
    model_dir = args.load_model_path.replace(ext, '')
    val_info = os.path.basename(os.path.dirname(args.val_list)) + '_' + os.path.basename(
        args.val_list.replace('.txt', ''))
    result_dir = os.path.join(model_dir, val_info + '_' + args.save_prefix)
    if not os.path.exists(result_dir):
        os.system('mkdir -p ' + result_dir)
    args.result_dir = result_dir


def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))


def prepare_train_args():
    args = get_train_args()
    get_train_model_dir(args)
    save_args(args, args.model_dir)
    return args


def prepare_test_args():
    args = get_test_args()
    get_test_result_dir(args)
    save_args(args, args.result_dir)
    return args


if __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args()
