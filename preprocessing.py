import os
import random
import shutil

# 원본 이미지가 있는 디렉토리와 대상 디렉토리를 지정합니다.
src_dir = "datasets/FFHQ/FFHQ"
dst_dir = "datasets/FFHQ2/img"

# 원본 디렉토리에서 모든 파일의 이름을 가져옵니다.
all_files = os.listdir(src_dir)

# 14,000개의 파일 이름을 무작위로 선택합니다.
selected_files = random.sample(all_files, 14000)

# 선택된 파일들을 새로운 디렉토리로 복사하면서 이름을 변경합니다.
for idx, filename in enumerate(selected_files):
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, f"{idx}.png")
    shutil.copy(src_path, dst_path)
