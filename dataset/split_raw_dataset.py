import os
import random
import shutil

# 设置原始文件夹和目标文件夹的路径
raw_folder = './raw-img'
output_folder = './cnn_dataset'

animal_class = ['cat', 'dog', 'chicken', 'horse', 'cow']

# 创建目标文件夹
os.makedirs(output_folder, exist_ok = True)

# 创建 train、test、valid 三个文件夹
for folder in ['train', 'test', 'valid']:
    os.makedirs(os.path.join(output_folder, folder), exist_ok = True)
    for subfolder in animal_class:
        os.makedirs(os.path.join(output_folder, folder, subfolder), exist_ok = True)

# 获取 raw 目录下的部分文件
raw_files = {}
for subdir in os.listdir(os.path.join(raw_folder)):
    if (subdir in animal_class):
        raw_files[subdir] = os.listdir(os.path.join(raw_folder, subdir))

# 将每个文件夹下的图片进行随机分配到 train、test、valid 中
for subdir in raw_files:
    random.shuffle(raw_files[subdir])
    num_files = len(raw_files[subdir])
    split1 = int(0.6 * num_files)  # 60% for training
    split2 = int(0.8 * num_files)  # 20% for testing, 20% for validation

    for idx, file in enumerate(raw_files[subdir]):
        source_path = os.path.join(raw_folder, subdir, file)

        if idx < split1:
            dest_folder = 'train'
        elif split1 <= idx < split2:
            dest_folder = 'test'
        else:
            dest_folder = 'valid'

        dest_path = os.path.join(output_folder, dest_folder, subdir, f"{idx}.jpg")
        shutil.copyfile(source_path, dest_path)

print("Spliting raw-img completed.")