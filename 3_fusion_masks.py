import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='融合伪标签和原始标签')
    parser.add_argument('--pred-dir', type=str, required=True,
                        help='模型预测的伪标签目录')
    parser.add_argument('--isaid-dir', type=str, required=True,
                        help='iSAID 原始标签目录')
    parser.add_argument('--merged-dir', type=str, required=True,
                        help='融合标签输出目录')
    return parser.parse_args()

def main():
    args = parse_args()
    
    pred_dir = args.pred_dir
    isaid_dir = args.isaid_dir
    merged_dir = args.merged_dir
    
    os.makedirs(merged_dir, exist_ok=True)

    # 获取文件列表
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".png")])
    isaid_files = sorted([f for f in os.listdir(isaid_dir) if f.endswith(".png")])

    print(f"伪标签文件: {len(pred_files)}, iSAID 标签文件: {len(isaid_files)}")
    print("开始融合...\n")

    # 统计信息
    stats = {"processed": 0, "skipped": 0, "shape_mismatch": 0}

    for fname_pred, fname_isaid in tqdm(zip(pred_files, isaid_files), total=len(pred_files)):

        # 文件名必须一致，否则跳过
        if fname_pred != fname_isaid:
            stats["skipped"] += 1
            continue

        pred_path = os.path.join(pred_dir, fname_pred)
        isaid_path = os.path.join(isaid_dir, fname_isaid)

        # 读取伪标签
        pred_mask = np.array(Image.open(pred_path))

        # 映射伪标签: 1-8 → 16-23（+15）
        mapped_pred = pred_mask.copy()
        cond_1_6 = (pred_mask >= 0) & (pred_mask <= 6)
        mapped_pred[cond_1_6] = pred_mask[cond_1_6] + 15

        # 读取 iSAID 标签（小目标 0-14）
        isaid_label = np.array(Image.open(isaid_path))

        # 尺寸检查
        if mapped_pred.shape != isaid_label.shape:
            stats["shape_mismatch"] += 1
            continue

        # 融合：iSAID 有效标签（0-14）和忽略区域（255）覆盖伪标签
        overwrite_mask = ((isaid_label >= 0) & (isaid_label <= 14)) | (isaid_label == 255)
        merged = mapped_pred.copy()
        merged[overwrite_mask] = isaid_label[overwrite_mask]

        # 保存结果
        Image.fromarray(merged).save(os.path.join(merged_dir, fname_pred))
        stats["processed"] += 1

    # 打印汇总
    print(f"\n=== 融合完成 ===")
    print(f"处理成功: {stats['processed']}")
    print(f"跳过(文件名不匹配): {stats['skipped']}")
    print(f"跳过(尺寸不匹配): {stats['shape_mismatch']}")
    print(f"输出目录: {merged_dir}")


if __name__ == '__main__':
    main()