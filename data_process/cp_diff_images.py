"""
Copy Different Mask Images
用于将差异评估结果中的mask文件复制到指定目录

根据 evaluate_diff.py 生成的 diff.txt 文件，将差异较大的样本的
old_mask 和 new_mask 复制到 exp/diff/ 目录下对应文件夹中
"""

import argparse
import os
import shutil
from tqdm import tqdm
import yaml

def parse_diff_line(line):
    """
    解析diff.txt的一行

    格式: image_path old_mask_path new_mask_path diff_ratio

    Args:
        line: 一行文本

    Returns:
        dict: 包含 image_path, old_mask_path, new_mask_path, diff_ratio
    """
    parts = line.strip().split()
    if len(parts) < 4:
        return None

    return {
        "image_path": parts[0],
        "old_mask_path": parts[1],
        "new_mask_path": parts[2],
        "diff_ratio": float(parts[3]),
    }


def copy_diff_masks(diff_file, data_root, output_dir):
    """
    复制差异mask文件到指定目录

    Args:
        diff_file: diff.txt文件路径
        data_root: 数据集根目录
        output_dir: 输出目录

    Returns:
        dict: 包含成功和失败的统计信息
    """
    # 创建输出目录
    old_mask_dir = os.path.join(output_dir, "old_mask")
    new_mask_dir = os.path.join(output_dir, "new_mask")

    os.makedirs(old_mask_dir, exist_ok=True)
    os.makedirs(new_mask_dir, exist_ok=True)

    # 读取diff文件
    if not os.path.exists(diff_file):
        raise FileNotFoundError(f"Diff文件不存在: {diff_file}")

    with open(diff_file, "r") as f:
        lines = f.readlines()

    print(f"找到 {len(lines)} 个差异样本")
    print(f"数据根目录: {data_root}")
    print(f"输出目录: {output_dir}")
    print()

    # 统计信息
    stats = {
        "total": len(lines),
        "success": 0,
        "failed": 0,
        "failed_files": [],
    }

    # 复制文件
    for line in tqdm(lines, desc="复制文件"):
        parsed = parse_diff_line(line)
        if parsed is None:
            continue

        # 复制 old_mask
        old_mask_src = os.path.join(data_root, parsed["old_mask_path"])
        old_mask_dst = os.path.join(old_mask_dir, os.path.basename(parsed["old_mask_path"]))

        # 复制 new_mask
        new_mask_src = os.path.join(data_root, parsed["new_mask_path"])
        new_mask_dst = os.path.join(new_mask_dir, os.path.basename(parsed["new_mask_path"]))

        # 执行复制
        success = True
        for src, dst, mask_type in [
            (old_mask_src, old_mask_dst, "old_mask"),
            (new_mask_src, new_mask_dst, "new_mask"),
        ]:
            if not os.path.exists(src):
                stats["failed_files"].append(
                    f"{mask_type}: {src} (源文件不存在)"
                )
                success = False
            else:
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    stats["failed_files"].append(
                        f"{mask_type}: {src} -> {dst} (错误: {str(e)})"
                    )
                    success = False

        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="复制差异mask文件 - 将差异较大的样本的mask复制到指定目录"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/MOTA.yaml",
    )
    parser.add_argument(
        "--diff-file",
        type=str,
        default=None,
        help="diff.txt文件路径 (默认: exp/diff/{dataset}_high_diff.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/diff/",
        help="输出目录 (默认: exp/diff/)",
    )

    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    args.dataset = cfg['dataset']
    args.data_root = cfg['data_root']
    # 确定diff文件路径
    if args.diff_file is None:
        args.diff_file = os.path.join(
            args.output_dir, f"{args.dataset}_high_diff.txt"
        )

    print("=" * 50)
    print("开始复制差异mask文件")
    print("=" * 50)

    try:
        stats = copy_diff_masks(args.diff_file, args.data_root, args.output_dir)

        # 显示统计信息
        print()
        print("=" * 50)
        print("复制完成！")
        print("=" * 50)
        print(f"总样本数: {stats['total']}")
        print(f"成功复制: {stats['success']}")
        print(f"失败数量: {stats['failed']}")
        print(f"old_mask目录: {os.path.join(args.output_dir, 'old_mask')}")
        print(f"new_mask目录: {os.path.join(args.output_dir, 'new_mask')}")

        if stats["failed"] > 0:
            print()
            print("失败的文件:")
            for failed_file in stats["failed_files"]:
                print(f"  - {failed_file}")

        print("=" * 50)

    except Exception as e:
        print()
        print("=" * 50)
        print("错误！")
        print("=" * 50)
        print(f"错误信息: {str(e)}")
        print("=" * 50)
        exit(1)


if __name__ == "__main__":
    main()