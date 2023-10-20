import torch
from torchviz import make_dot
from modules import UNet

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # GPU 사용 가능한 경우 "cuda:0", 그렇지 않은 경우 "cpu"

# 더미 입력 데이터
batch_size = 1
x = torch.randn(batch_size, 3, 256, 256).to(device)
t = torch.randn(batch_size).to(device).type(torch.float)

# 모델 초기화
model = UNet().to(device)

# 모델 forward pass 실행
output = model(x, t)

# 계산 그래프 시각화
dot = make_dot(output)
dot.format = 'png'
dot.render('unet_graph')
