import os
from PIL import Image
import numpy as np
from tqdm import tqdm

open_dir = "/data/users/lanjie/Project/S5_finetune/exp/visualizations/predictions"  # 模型预测
mask_dir = "/data/users/lanjie/dataset/MOTA/split_ms_dota1_0/train/new_labels_255"  #原始标签
merged_dir = "/data/users/lanjie/dataset/MOTA/split_ms_dota1_0/train/merged_IRSAMap_masks_2/" #融合标签

os.makedirs(merged_dir, exist_ok=True)

# 排序，确保文件一一对应
open_files = sorted([f for f in os.listdir(open_dir) if f.endswith(".png")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

print(f"Found {len(open_files)} open_masks, {len(mask_files)} masks")
print("Start merging...\n")

for fname_open, fname_mask in tqdm(zip(open_files, mask_files), total=len(open_files)):

    # 文件名必须一致，否则跳过（保护性检查）
    if fname_open != fname_mask:
        print(f"[WARNING] File mismatch: {fname_open} vs {fname_mask}, skipped.")
        continue

    open_path = os.path.join(open_dir, fname_open)
    mask_path = os.path.join(mask_dir, fname_mask)

    # 读取 open_mask
    open_mask = np.array(Image.open(open_path))

    # # 映射 open_mask 0-8 → 15-23
    # mapped_open = open_mask.copy()
    # mapped_open = open_mask + 15


    mapped_open = open_mask.copy()

    # 1–8 → 16–23（+15）
    cond_1_6 = (open_mask >= 0) & (open_mask <= 6)
    mapped_open[cond_1_6] = open_mask[cond_1_6] + 15
    # 读取 masks（小目标 0-14）
    mask = np.array(Image.open(mask_path))

    # 尺寸检查
    if mapped_open.shape != mask.shape:
        print(f"[WARNING] Shape mismatch for {fname_open}: open {mapped_open.shape}, mask {mask.shape}")
        continue

    # 打印合并前标签信息
    print(f"\n[{fname_open}] Before merge:")
    print(f"  open_mask labels       : {np.unique(open_mask)}")
    print(f"  mapped_open labels     : {np.unique(mapped_open)}")
    print(f"  small-object mask labels: {np.unique(mask)}")

    # 初始化 merged
    overwrite_mask = ((mask >= 0) & (mask <= 14)) | (mask == 255)

    merged = mapped_open.copy()
    merged[overwrite_mask] = mask[overwrite_mask]

    # 打印合并后的标签信息
    print(f"  merged labels          : {np.unique(merged)}\n")

    # 保存结果
    Image.fromarray(merged).save(os.path.join(merged_dir, fname_open))

print("\nAll done! Merged masks saved to:", merged_dir)
