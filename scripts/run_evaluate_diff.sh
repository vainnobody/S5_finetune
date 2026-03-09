#!/bin/bash

# S5_finetune 标签差异评估运行脚本
# 用于对比两种模型生成的标签，找出差异最大的样本

# 默认参数
CONFIG=${CONFIG:-"configs/MOTA.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"exp/diff/"}
THRESHOLD=${THRESHOLD:-0.03}
OUTPUT_ALL=${OUTPUT_ALL:-false}

# 打印使用说明
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --config CONFIG_PATH       配置文件路径 (默认: configs/MOTA.yaml)"
    echo "                              支持的配置文件: loveda.yaml, isaid_ori.yaml, potsdam.yaml 等"
    echo "  --output-dir DIR_PATH      输出结果保存目录 (默认: exp/diff/)"
    echo "  --threshold THRESHOLD      差异率阈值，筛选差值小于等于此阈值的样本 (默认: 0.01)"
    echo "                              取值范围: 0.0 ~ 1.0"
    echo "  --output-all               输出所有样本的差异（按差异排序）"
    echo "  -h, --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 基本使用 (使用 MOTA 配置，阈值 0.01)"
    echo "  $0 --config configs/MOTA.yaml"
    echo ""
    echo "  # 指定输出目录"
    echo "  $0 --config configs/isaid_ori.yaml --output-dir exp/isaid_diff/"
    echo ""
    echo "  # 使用更严格的阈值 (0.005)"
    echo "  $0 --config configs/potsdam.yaml --threshold 0.005"
    echo ""
    echo "  # 输出所有样本的差异"
    echo "  $0 --config configs/vaihingen.yaml --output-all"
    echo ""
    echo "输出格式:"
    echo "  输出文件: {output_dir}/{dataset}_diff.txt"
    echo "  格式: image_id diff_ratio"
    echo ""
    echo "  其中 diff_ratio 为像素不一致率 (0.0 ~ 1.0)"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --output-all)
            OUTPUT_ALL=true
            shift 1
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
if [ -z "$CONFIG" ]; then
    echo "错误: 必须指定配置文件 (--config)"
    echo ""
    usage
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 创建输出目录
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# 打印运行信息
echo "========================================"
echo "开始标签差异评估"
echo "========================================"
echo "配置文件:     $CONFIG"
echo "输出目录:     $OUTPUT_DIR"
echo "差异阈值:     $THRESHOLD"
echo "输出全部:     $OUTPUT_ALL"
echo "========================================"

# 构建命令
CMD="python evaluate_diff.py --config \"$CONFIG\" --output-dir \"$OUTPUT_DIR\" --threshold $THRESHOLD"

if [ "$OUTPUT_ALL" = true ]; then
    CMD="$CMD --output-all"
fi

# 运行评估脚本
eval $CMD

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "标签差异评估完成！"
    echo "========================================"
    echo "结果保存在: $OUTPUT_DIR"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "标签差异评估失败！"
    echo "========================================"
    exit 1
fi
