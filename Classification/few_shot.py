import torch.utils.data as data

from PIL import Image
import os
import os.path
from torchvision import transforms
from torchvision.transforms import InterpolationMode, CenterCrop
import torch
from timm.data import create_transform
    
def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append(("images/" + impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def get_few_shot(name, batch_size, prefix, evaluate=False):
    root = './data/few-shot/' + name
    transform_train = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )

    transform_test = transforms.Compose([
            #transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if evaluate:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/annotations/train_meta.list." + prefix,
                transform=transform_train),
            batch_size=batch_size, shuffle=False, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/annotations/val_meta.list",
                transform=transform_test),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/annotations/train_meta.list." + prefix,
                transform=transform_train),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root, flist=root + "/annotations/test_meta.list",
                transform=transform_test),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
    return train_loader, val_loader
