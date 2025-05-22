from tqdm import tqdm
from avalanche.evaluation.metrics.accuracy import Accuracy
from options import prepare_train_args
from utils.classification_utils import *
from vtab import *
from torchinfo import summary
from utils.torch_utils import load_match_dict, load_dict
from model.model_entry import select_model


##Evaluation (Validation)
def test(model, dl, incorrect_file=None):
    model.eval()
    loop = tqdm(dl, desc='Val:')
    acc = Accuracy()
    model.to(device)
    for i, data in enumerate(loop):
        img, label = data[0].to(device), data[1].to(device)
        output = model(img).data
        acc.update(output.argmax(dim=1).view(-1), label)
        if incorrect_file is not None:
            for q in range(len(label)):
                if output.argmax(dim=1)[q] != label[q]:
                    img_tensor = dl.dataset.__getitem__(q)[0]
                    img = transforms.ToPILImage(mode='RGB')(img_tensor)
                    img_path = os.path.join(incorrect_file, str(q) + '.jpg')
                    print(img_path)
                    img.save(img_path)
    print('acc:', acc.result())
    return acc.result()


if __name__ == '__main__':
    args = prepare_train_args()  # 初始化命令行参数
    set_seed(args.seed)

    config = get_config(args.data_type)
    gpus = args.gpus
    device = torch.device(gpus)
    train_loader, test_loader = get_data(args.data_type, args.batch_size)

    model = select_model(args, config)
    # 加载模型（load_model_path有权重时）
    if args.load_model_path != '':
        print("=> using pre-trained weights for DPSNet")
        if args.load_not_strict:
            load_match_dict(model, args.load_model_path)
        else:
            model.load_state_dict(torch.load(args.load_model_path).state_dict())
    else:
        load_dict(model, config['class_num'])
    summary(model, input_size=(1, 3, 224, 224))

    acc = test(model, test_loader)
    print('Accuracy:', acc)
    with open(args.model_dir + args.data_type + '.log', 'w') as f:
        f.write(str(acc))
