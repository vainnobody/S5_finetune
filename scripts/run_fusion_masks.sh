#!/bin/bash

# S5_finetune 标签融合启动脚本
# 用于融合模型预测的伪标签和原始标签

# 默认参数
PRED_DIR=${PRED_DIR:-""}
ISAID_DIR=${ISAID_DIR:-""}
MERGED_DIR=${MERGED_DIR:-""}

# 打印使用说明
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --pred-dir DIR      模型预测的伪标签目录 (必需)"
    echo "  --isaid-dir DIR     iSAID 原始标签目录 (必需)"
    echo "  --merged-dir DIR    融合标签输出目录 (必需)"
    echo "  -h, --help          显示此帮助信息"
    echo ""
    echo "说明:"
    echo "  该脚本用于融合模型预测的伪标签和原始标签。"
    echo "  融合规则:"
    echo "    - 伪标签类别 0-6 映射为 15-21 (+15)"
    echo "    - 原始标签有效区域 (0-14) 和忽略区域 (255) 覆盖伪标签"
    echo ""
    echo "示例:"
    echo "  # 基本使用"
    echo "  $0 --pred-dir exp/inference/predictions \\"
    echo "     --isaid-dir /path/to/isaid/labels \\"
    echo "     --merged-dir exp/merged_masks"
    echo ""
    echo "  # 使用环境变量"
    echo "  PRED_DIR=exp/preds ISAID_DIR=/data/labels MERGED_DIR=exp/merged $0"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --pred-dir)
            PRED_DIR="$2"
            shift 2
            ;;
        --isaid-dir)
            ISAID_DIR="$2"
            shift 2
            ;;
        --merged-dir)
            MERGED_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            usage
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$PRED_DIR" ]; then
    echo "错误: 必须指定伪标签目录 (--pred-dir)"
    usage
    exit 1
fi

if [ -z "$ISAID_DIR" ]; then
    echo "错误: 必须指定原始标签目录 (--isaid-dir)"
    usage
    exit 1
fi

if [ -z "$MERGED_DIR" ]; then
    echo "错误: 必须指定融合输出目录 (--merged-dir)"
    usage
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$PRED_DIR" ]; then
    echo "错误: 伪标签目录不存在: $PRED_DIR"
    exit 1
fi

if [ ! -d "$ISAID_DIR" ]; then
    echo "错误: 原始标签目录不存在: $ISAID_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$MERGED_DIR"

# 打印运行信息
echo "========================================"
echo "开始标签融合"
echo "========================================"
echo "伪标签目录:   $PRED_DIR"
echo "原始标签目录: $ISAID_DIR"
echo "融合输出目录: $MERGED_DIR"
echo "========================================"

# 运行融合脚本
python 3_fusion_masks.py \
    --pred-dir "$PRED_DIR" \
    --isaid-dir "$ISAID_DIR" \
    --merged-dir "$MERGED_DIR"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "标签融合完成！"
    echo "========================================"
    echo "融合标签保存在: $MERGED_DIR"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "标签融合失败！"
    echo "========================================"
    exit 1
fi
