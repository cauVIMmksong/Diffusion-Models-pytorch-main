import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from utils import get_data
import argparse
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm

# Arguments
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1  # Modify this as per your needs
args.image_size = 64
args.dataset_path = r"datasets/FFHQ"

# Create DataLoader
dataloader = get_data(args)

# Create Diffusion instance
diff = Diffusion(device="cuda")

# Sample an image from dataset
image = next(iter(dataloader))[0]

# Generate and save an initial noised image
t = torch.Tensor([0, 50, 100, 150, 300, 600, 700, 999]).long()
noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), "noise2.jpg")

def extract_principal_components(dataset, n_components=5, sample_fraction=0.1):
    # Use tqdm to show the progress
    # 데이터셋의 일부만 선택
    num_samples = int(len(dataset) * sample_fraction)
    sampled_indices = np.random.choice(len(dataset), num_samples, replace=False)
    sampled_images = [dataset[i][0].numpy() for i in tqdm(sampled_indices, desc="Extracting PCA")]

    sampled_images = np.stack(sampled_images)

    means = []
    for channel in range(3):
        pca = PCA(n_components=n_components)
        channel_data = sampled_images[:, channel, :, :].reshape(sampled_images.shape[0], -1)
        pca.fit(channel_data)
        means.append(pca.mean_)

    mean_image = np.stack(means).reshape(3, args.image_size, args.image_size)
    return torch.tensor(mean_image)

def visualize_pca_features(pca_features):
    """
    Visualize the PCA features extracted from the dataset.

    Parameters:
    - pca_features: PCA features of shape (num_components, image_height, image_width)
    """
    num_components = pca_features.shape[0]

    plt.figure(figsize=(15, 5))
    for i in range(num_components):
        plt.subplot(1, num_components, i+1)
        plt.imshow(pca_features[i], cmap='gray')
        plt.title(f'Component {i+1}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Add principal component to noise at specified timesteps
timesteps_to_modify = [0, 100, 500, 800, 999]
modified_noised_images = []

principal_component = extract_principal_components(dataloader.dataset)
principal_component = principal_component.to("cuda")
visualize_pca_features(principal_component.cpu().numpy())


for t_val in tqdm(timesteps_to_modify, desc="Modifying noise"):
    img, _ = diff.noise_images(image, torch.tensor([t_val]))
    img += principal_component
    modified_noised_images.append(img)

# Clamp image values between [0,1]
noised_images = [img.clamp(0, 1) for img in modified_noised_images]
for img in modified_noised_images:
    print(img.shape)

corrected_noised_images = [img.squeeze(0) for img in modified_noised_images]
noised_images_tensor = torch.stack(corrected_noised_images)

# Save the visualized images
save_image(noised_images_tensor, "modified_noise.jpg")
