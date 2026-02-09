import yaml
import argparse
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from dataset.label_compare import LabelCompareDataset


def evaluate(loader, cfg):
    """评估标签差异"""
    results = []

    for img_path, old_mask, new_mask, id in tqdm(loader, desc="计算差异"):
        # DataLoader 返回的是 batch，取第一个元素
        img_path = img_path[0]
        old_mask = old_mask[0].numpy()
        new_mask = new_mask[0].numpy()
        id = id[0]

        # 计算差异率
        ignore_value = cfg.get("ignore_value", 255)
        valid_mask = (old_mask != ignore_value) & (new_mask != ignore_value)
        if valid_mask.sum() == 0:
            diff_ratio = 0.0
        else:
            diff = (old_mask != new_mask) & valid_mask
            diff_ratio = float(diff.sum()) / float(valid_mask.sum())

        results.append(
            {
                "id": id,
                "img_path": img_path,
                "diff_ratio": diff_ratio,
            }
        )

    # 按差异率降序排序
    results.sort(key=lambda x: x["diff_ratio"], reverse=True)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Label Differences - 评估两种标签的差异"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/loveda.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/diff/",
        help="输出结果保存目录",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=1.0,
        help="输出差异最大的前百分之N的样本 (默认: 1.0)",
    )
    parser.add_argument(
        "--output-all",
        action="store_true",
        help="输出所有样本的差异（按差异排序）",
    )

    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg["output_dir"] = args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = LabelCompareDataset(cfg["dataset"], cfg["data_root"], "compare")

    loader = DataLoader(
        dataset, batch_size=1, pin_memory=True, num_workers=8, drop_last=False
    )
    print("Total samples: {}\n".format(len(dataset)))

    results = evaluate(loader, cfg)

    # 计算统计信息
    diff_ratios = [r["diff_ratio"] for r in results]
    mean_diff = sum(diff_ratios) / len(diff_ratios) if diff_ratios else 0
    max_diff = max(diff_ratios) if diff_ratios else 0
    min_diff = min(diff_ratios) if diff_ratios else 0

    print("***** Statistics *****")
    print("Mean diff: {:.4f} ({:.2f}%)".format(mean_diff, mean_diff * 100))
    print("Max diff: {:.4f} ({:.2f}%)".format(max_diff, max_diff * 100))
    print("Min diff: {:.4f} ({:.2f}%)\n".format(min_diff, min_diff * 100))

    # 确定输出数量
    if args.output_all:
        output_count = len(results)
    else:
        output_count = max(1, int(len(results) * args.top_percent / 100))

    print("Output {} samples with highest diff...\n".format(output_count))

    # 输出文件路径
    output_path = os.path.join(
        args.output_dir, "{}_high_diff.txt".format(cfg["dataset"])
    )

    # 写入结果 (格式: image_path old_mask_path new_mask_path diff_ratio)
    with open(output_path, "w") as f:
        for r in results[:output_count]:
            line = "{} {:.6f}\n".format(r["id"], r["diff_ratio"])
            f.write(line)

    print("Results saved to: {}\n".format(output_path))

    # 显示前10个差异最大的样本
    print("***** Top 10 High Diff Samples *****")
    for i, r in enumerate(results[:10]):
        print("{:2d}. {}: {:.2f}%".format(i + 1, r["img_path"], r["diff_ratio"] * 100))


if __name__ == "__main__":
    main()
