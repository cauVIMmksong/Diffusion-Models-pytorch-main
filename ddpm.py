import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
from torchvision.utils import make_grid
import logging
import time

import random
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# Add PCA feature extraction function
from sklearn.decomposition import PCA

def extract_pca_features(dataset, n_components=3):
    images = [np.array(img[0]).reshape(-1) for img in dataset]
    images = np.array(images)
    pca = PCA(n_components=n_components)
    pca.fit(images)
    return pca.components_

class Diffusion:
    def __init__(self, pca_features=None, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.pca_features = pca_features

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        Ɛ = torch.randn_like(x)
        if (t % 100).any() and self.pca_features is not None:
            feature_noise = torch.tensor(self.pca_features).float().to(self.device)
            Ɛ *= feature_noise  # Apply PCA features to noise
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    pca_features = extract_pca_features(dataloader.dataset)
    diffusion = Diffusion(pca_features=pca_features, img_size=args.image_size, device=device)
    
    losses = []  # MSE losses for each epoch
    start_time = time.time()

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        epoch_losses = []  # Store losses for this epoch
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(MSE=loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        logging.info(f"Epoch {epoch} average MSE: {avg_loss:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Training finished in {elapsed_time:.2f} seconds.")
    
    # Plotting the MSE losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="MSE Loss", color="blue")
    plt.title("Training MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("results", args.run_name, "mse_loss_plot.png"))
    plt.show()

    return model, diffusion


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "PCA_DDPM_FFHQ"
    args.epochs = 100
    args.batch_size = 4
    args.image_size = 64
    args.dataset_path = r"datasets/FFHQ"
    args.device = "cuda"
    args.lr = 3e-4
    model, diffusion = train(args)
    
    return model, diffusion
    
def sample_images(dataset_path, output_path, grid_size=8):
    # 데이터셋 불러오기 및 전처리
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        #transforms.CenterCrop((224, 224)),
        transforms.ToTensor()
    ])
    # 데이터셋 불러오기
    dataset = dset.ImageFolder(dataset_path, transform=transform)
    
    # 랜덤으로 64개의 이미지 선택
    images = []
    for i in range(grid_size**2):
        index = random.randint(0, len(dataset) - 1)
        image, _ = dataset[index]
        images.append(image)
    
    # 이미지 그리드 생성
    grid = make_grid(images, nrow=grid_size, pad_value=1)

    # 이미지 저장
    os.makedirs(output_path, exist_ok=True)
    plt.figure(figsize=(16, 16))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(os.path.join(output_path, 'sample.png'))

#두줄 샘플링
if __name__ == '__main__':
    model, diffusion = launch()

    # 학습 완료 후 샘플 이미지 생성 및 모델 저장
    sampled_images = diffusion.sample(model, n=args.batch_size)  # batch_size 대신 원하는 샘플 이미지의 개수를 지정할 수 있습니다.
    save_images(sampled_images, os.path.join("results", args.run_name, f"final_samples.jpg"))
    torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))    

    dataset_path = 'datasets/FFHQ/FFHQ'
    output_path = 'results/PCA_DDPM_FFHQ'
    sample_images(dataset_path, output_path, grid_size=8)
    
    device = "cuda"
    #model = UNet().to(device)
    ckpt = torch.load("models\PCA_DDPM_FFHQ\ckpt.pt")
    model.load_state_dict(ckpt)
    #diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, 10)
    print(x.shape)

    x_grid = make_grid(x, nrow=4, pad_value=1)
    x_grid_np = (x_grid.permute(1, 2, 0)*255).type(torch.uint8).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(x_grid_np)
    ax.set_title("Diffusion Sampling Results", fontsize=16)
    plt.show()

#원본데이터 샘플링 파일
    
""" 
if __name__ == '__main__':
    launch()
    
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("models\DDPM_Uncondtional\ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, 8)
    print(x.shape)
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
         torch.cat([i for i in x.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()
 """

# %%
