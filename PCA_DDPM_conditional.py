import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torchvision.utils import save_image
from utils import *
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# Add PCA feature extraction function
def extract_pca_features_by_class(dataset, n_components=1):
    class_images = {i: [] for i in range(10)}  # CIFAR10의 클래스가 10개라고 가정

    for img, label in dataset:
        class_images[label].append(img)

    class_pca_features = {}

    for label, images in class_images.items():
        pca_features = []
        for channel in range(3):  # RGB channels
            channel_data = [np.array(img[channel]).reshape(-1) for img in images]
            channel_data = np.array(channel_data)

            pca = PCA(n_components=n_components)
            pca.fit(channel_data)

            # 첫 번째 주성분만 선택합니다.
            img_height = images[0].shape[1]
            img_width = images[0].shape[2]
            pca_features.append(pca.components_[0].reshape(1, img_height, img_width))

        class_pca_features[label] = np.array(pca_features)

    return class_pca_features


class Diffusion:
    def __init__(self, pca_features_per_class=None, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.pca_features_per_class = pca_features_per_class
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def apply_pca_features(self, x, labels):
        n = x.shape[0]
        n_pca_applied = int(n * 0.10)  # 10%의 이미지 수

        # 적용할 이미지의 인덱스를 무작위로 선택
        indices_to_apply = np.random.choice(n, n_pca_applied, replace=False)

        for i in indices_to_apply:
            label = labels[i]
            pca_features_for_label = torch.tensor(self.pca_features_per_class[label.item()]).to(self.device).float()
            for c in range(3):  # RGB channels
                x[i, c] += pca_features_for_label[c].squeeze(0)

        return x



    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            x = self.apply_pca_features(x, labels)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
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
    pca_features_per_class = extract_pca_features_by_class(dataloader.dataset)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(pca_features_per_class=pca_features_per_class, img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0 or epoch == (args.epochs - 1):
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_{epoch}.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "PCA_DDPM_CIFAR10(v3)"
    args.epochs = 100
    args.batch_size = 10
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = r"datasets/cifar10-32/train"
    args.device = "cuda"
    args.lr = 3e-4
#    train(args)

# 기존 샘플링 코드
if __name__ == '__main__':
    launch()

    # args 객체 생성
    class Args:
        def __init__(self):
            self.run_name = "PCA_DDPM_CIFAR10(v3)"
            self.batch_size = 10
            self.image_size = 64
            self.num_classes = 10
            self.dataset_path = r"datasets/cifar10-32/train"
            self.device = "cuda"
            self.lr = 3e-4

    args = Args()  # args 객체 인스턴스화

    device = "cuda"
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load("./models/PCA_DDPM_CIFAR10(v3)/ckpt_99.pt")
    model.load_state_dict(ckpt)

     # 아래 두 줄을 추가합니다.
    dataloader = get_data(args)  # 데이터 로더 생성
    pca_features_per_class = extract_pca_features_by_class(dataloader.dataset)  # PCA 특징 추출
    
    # pca_features_per_class 인자를 추가하여 Diffusion 객체를 생성합니다.
    diffusion = Diffusion(pca_features_per_class=pca_features_per_class, img_size=64, device=device)

    """n = 16
    y = torch.Tensor([6] * n).long().to(device)
    x = diffusion.sample(model, n, y, cfg_scale=0)
    plot_images(x)
    # 이미지 저장
    plt.savefig('results/PCA_DDPM_CIFAR10(v3)/sampled_images.png', bbox_inches='tight', pad_inches=0)"""

    # fake_CIFAR10_img/img 디렉토리 생성 (해당 디렉토리가 없는 경우에만)
    save_dir = 'fake_CIFAR10_img/img'
    os.makedirs(save_dir, exist_ok=True)
    
    for i in tqdm(range(6000), desc="Generating and Saving Images"):
        # 한 장씩 이미지를 샘플링
        y = torch.randint(0, 10, (1,)).long().to(device)  # 0-9 사이의 레이블을 무작위로 선택
        x = diffusion.sample(model, 1, y, cfg_scale=0)
        x = (x.float() / 255.0).clamp(0, 1)
        
        # 이미지를 fake_CIFAR10_img/img 디렉토리에 저장
        filename = os.path.join(save_dir, f"fake_image_{i}.png")
        save_image(x[0], filename)