import numpy as np
import torch


def psnr(img_pred, img_true, max_val=1.0):
    assert img_true.shape == img_pred.shape, (
        f'Image shapes are differnet: {img_true.shape}, {img_pred.shape}.')
    batch_size = img_true.size(0)
    mse = torch.mean((img_true - img_pred) ** 2, axis=(1, 2, 3))
    psnr = torch.zeros(batch_size)
    for i in range(batch_size):
        psnr[i] = 10 * torch.log10(max_val ** 2 / mse[i])
    avg_psnr = torch.mean(psnr)
    return avg_psnr.item()



