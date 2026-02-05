import os
import glob
from PIL import Image
import numpy as np
from collections import OrderedDict

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

# 构建 RGB 到类别索引的映射字典
rgb_to_class = {value: idx for idx, value in enumerate(MAPPING.values())}
idx_to_label = {idx: label for label, idx in rgb_to_class.items()}

def rgb_to_class_map(image):
    h, w, _ = image.shape
    # 创建新的图像来自类别索引
    class_image = np.zeros((h, w), dtype=np.int8)

    # 将每个像素的 RGB 颜色值映射到类别索引
    for rgb, cls in rgb_to_class.items():
        mask = (image == rgb).all(axis=2)
        class_image[mask] = cls

    # 处理(124, 116, 104)的像素，设为 background 类别
    background_rgb = (124, 116, 104)
    mask = (image == background_rgb).all(axis=2)
    class_image[mask] = len(MAPPING) - 1  # background 类别的默認索引

    return class_image

def convert_images_in_folder(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取文件夹下所有的图片文件
    image_files = glob.glob(os.path.join(input_folder, "*.png"))

    for image_file in image_files:
        # 打开图像并转换为 numpy 数组
        image = Image.open(image_file)
        image_array = np.array(image)

        # 转换 RGB 图为类标图
        class_image = rgb_to_class_map(image_array)

        # 转换类别标签图为 PIL 图像
        class_image_pil = Image.fromarray(class_image.astype(np.uint8))

        # 构建输出路径
        base_name = os.path.basename(image_file)
        output_path = os.path.join(output_folder, base_name)

        # 保存图像
        class_image_pil.save(output_path)

        print(f"Saved mapped image: {output_path}")

# 设置输入和输出文件夹路径
input_folder = "/data1/users/zhengzhiyu/mtp_workplace/dataset/MOTA/split_ms_dota1_0/train/mask/"
output_folder = "/data1/users/zhengzhiyu/mtp_workplace/dataset/MOTA/split_ms_dota1_0/train/matching_det_seg_labels/"

# 转换并保存图像
convert_images_in_folder(input_folder, output_folder)
