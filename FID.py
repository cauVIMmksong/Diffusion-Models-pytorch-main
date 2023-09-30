from torchvision.models import inception_v3
from scipy.linalg import sqrtm


# Inception 모델 로딩
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()

def get_features(data_loader, model):
    all_features = []
    for images, _ in data_loader:
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

# 몇 개의 샘플 이미지를 시각화합니다.
sample_batch = next(iter(real_dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Sample Images")
plt.imshow(np.transpose(vutils.make_grid(sample_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

# 1. 실제 데이터셋에 대한 Dataloader 생성
transform = T.Compose([
    T.Resize((299,299)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

real_dataset = dset.CIFAR10(root='./data', train=True, transform=transform, download=True)
real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

# 2. 가짜 데이터셋에 대한 Dataloader 생성
fake_dataset = dset.ImageFolder(root = 'fake_image_CIFAR10', transform=transform)
fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)

# 실제 데이터와 생성된 데이터의 특징 벡터를 추출
real_features = get_features(real_dataloader, inception_model)
fake_features = get_features(fake_dataloader, inception_model)

# FID 계산
fid_value = calculate_fid(real_features, fake_features)
print("FID:", fid_value)
