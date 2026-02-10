from collections import OrderedDict
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


MAPPING = OrderedDict({
    'large_Vehicle': (0, 127, 127),
    'swimming_pool': (0, 0, 255),
    'helicopter': (0, 0, 191),
    'bridge': (0, 127, 63),
    'plane': (0, 127, 255),
    'ship': (0, 0, 63),
    'soccer_ball_field': (0, 127, 191),
    'basketball_court': (0, 63, 191),
    'ground_Track_Field': (0, 63, 255),
    'small_Vehicle': (0, 0, 127),
    'baseball_diamond': (0, 63, 0),
    'tennis_court': (0, 63, 127),
    'roundabout': (0, 63, 63),
    'storage_tank': (0, 191, 127),
    'harbor': (0, 100, 155),
    'background': (108, 98, 101),
    'Cropland': (255, 253, 145),
    'Tree': (32, 216, 109),
    'Grass': (1, 252, 119),
    'Water': (20, 197, 232),
    'Buildings': (210, 75, 97),
    'Road': (255, 200, 1)
})


category_to_index = {category: idx for idx, category in enumerate(MAPPING.keys())}

# 将MAPPING变换为索引到RGB的映射
index_to_color = np.array(list(MAPPING.values()), dtype=np.uint8)

def convert_mask_to_rgb(mask_image_path, output_image_path):
    # 打开分割mask图像
    mask_image = Image.open(mask_image_path)
    
    # 将图像转换为numpy数组
    mask_array = np.array(mask_image)
    
    # 创建RGB图像的空数组
    rgb_array = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    
    # 将mask数组中的类别索引映射到RGB值
    for index, color in enumerate(index_to_color):
        rgb_array[mask_array == index] = color
    
    # 创建RGB图像
    rgb_image = Image.fromarray(rgb_array)
    
    # 保存RGB图像
    rgb_image.save(output_image_path)

def batch_convert_masks(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pbar = tqdm(os.listdir(input_folder))
    for filename in pbar:
        if filename.endswith(".png"):  # 假设所有mask图像都是.png格式
            mask_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            convert_mask_to_rgb(mask_image_path, output_image_path)
            # print(f"Processed {filename}")

# 使用示例，将'input_masks'文件夹中的所有文件转换并保存到'output_rgbs'文件夹中
# input_folder = "/data/users/lanjie/Project/S5_finetune/exp/diff/old_mask"
# output_folder = "/data/users/lanjie/Project/S5_finetune/exp/diff/old_mask_rgb"
input_folder = "/data/users/lanjie/Project/S5_finetune/exp/diff/new_mask"
output_folder = "/data/users/lanjie/Project/S5_finetune/exp/diff/new_mask_rgb"
batch_convert_masks(input_folder, output_folder)
