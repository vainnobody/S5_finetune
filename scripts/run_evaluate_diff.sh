#!/bin/bash

# S5_finetune 标签差异评估运行脚本
# 用于对比两种模型生成的标签，找出差异最大的样本

# 默认参数
INPUT=${INPUT:-""}
OUTPUT=${OUTPUT:-"exp/diff/high_diff.txt"}
DATA_ROOT=${DATA_ROOT:-""}
IGNORE_VALUE=${IGNORE_VALUE:-255}
TOP_PERCENT=${TOP_PERCENT:-1.0}
OUTPUT_ALL=${OUTPUT_ALL:-false}

# 打印使用说明
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --input INPUT_PATH          输入splits文件路径 (必需)"
    echo "                              格式: image_path old_mask_path new_mask_path"
    echo "  --output OUTPUT_PATH        输出结果文件路径 (默认: exp/diff/high_diff.txt)"
    echo "  --data-root ROOT_DIR        数据集根目录 (默认: 当前目录)"
    echo "  --ignore-value VALUE        忽略的标签值 (默认: 255)"
    echo "  --top-percent PERCENT       输出差异最大的前百分之N (默认: 1.0)"
    echo "  --output-all                输出所有样本的差异（按差异排序）"
    echo "  -h, --help                  显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 基本使用"
    echo "  $0 --input splits/loveda/compare.txt --output exp/diff/loveda_high_diff.txt"
    echo ""
    echo "  # 指定数据根目录"
    echo "  $0 --input splits/loveda/compare.txt --data-root /path/to/dataset"
    echo ""
    echo "  # 输出前5%的高差异样本"
    echo "  $0 --input splits/loveda/compare.txt --top-percent 5"
    echo ""
    echo "  # 输出所有样本的差异"
    echo "  $0 --input splits/loveda/compare.txt --output-all"
    echo ""
    echo "输出格式:"
    echo "  image_path old_mask_path new_mask_path diff_ratio"
    echo ""
    echo "  其中 diff_ratio 为像素不一致率 (0.0 ~ 1.0)"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --ignore-value)
            IGNORE_VALUE="$2"
            shift 2
            ;;
        --top-percent)
            TOP_PERCENT="$2"
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
if [ -z "$INPUT" ]; then
    echo "错误: 必须指定输入文件 (--input)"
    echo ""
    usage
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT" ]; then
    echo "错误: 输入文件不存在: $INPUT"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR=$(dirname "$OUTPUT")
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# 打印运行信息
echo "========================================"
echo "开始标签差异评估"
echo "========================================"
echo "输入文件:     $INPUT"
echo "输出文件:     $OUTPUT"
echo "数据根目录:   ${DATA_ROOT:-'(当前目录)'}"
echo "忽略值:       $IGNORE_VALUE"
echo "输出前:       ${TOP_PERCENT}%"
echo "输出全部:     $OUTPUT_ALL"
echo "========================================"

# 构建命令
CMD="python evaluate_diff.py --input \"$INPUT\" --output \"$OUTPUT\" --ignore-value $IGNORE_VALUE --top-percent $TOP_PERCENT"

if [ -n "$DATA_ROOT" ]; then
    CMD="$CMD --data-root \"$DATA_ROOT\""
fi

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
    echo "结果保存在: $OUTPUT"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "标签差异评估失败！"
    echo "========================================"
    exit 1
fi
