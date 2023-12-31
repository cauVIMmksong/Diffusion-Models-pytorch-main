#%%

import os
import random
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from tqdm import tqdm

#%%
ngpu = 1
batch_size = 128

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device,"is available!")

# Inception 모델 로딩
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()

def get_features(data_loader, model):
    all_features = []
    for images, _ in tqdm(data_loader, desc="Extracting Features"):  # tqdm 추가
        with torch.no_grad():
            features = model(images.to(device))
        all_features.append(features.cpu().numpy())
    return np.concatenate(all_features, axis=0)

# FID 계산
def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# 1. 실제 데이터셋에 대한 Dataloader 생성
transform = T.Compose([
    T.Resize((299,299)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

real_dataset = dset.ImageFolder(root='datasets/FFHQ2', transform=transform)
real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

# 2. 가짜 데이터셋에 대한 Dataloader 생성
fake_dataset = dset.ImageFolder(root = 'results/FFHQ_fake_3_img', transform=transform)
fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)

# 몇 개의 샘플 이미지를 시각화합니다.
sample_batch = next(iter(real_dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Sample Images")
plt.imshow(np.transpose(vutils.make_grid(sample_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

#%%

# 실제 데이터와 생성된 데이터의 특징 벡터를 추출
print("Extracting features from real data...")
real_features = get_features(real_dataloader, inception_model)
print("Extracting features from fake data...")
fake_features = get_features(fake_dataloader, inception_model)

# FID 계산
print("Calculating FID...")
fid_value = calculate_fid(real_features, fake_features)
print("FID:", fid_value)
