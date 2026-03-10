#!/bin/bash

# S5_finetune 模型训练启动脚本
# 用于分布式训练语义分割模型

# 默认参数
CONFIG=${CONFIG:-"configs/isaid_ori.yaml"}
LABELED_ID_PATH=${LABELED_ID_PATH:-"splits/isaid_ori/all/labeled.txt"}
UNLABELED_ID_PATH=${UNLABELED_ID_PATH:-""}
SAVE_PATH=${SAVE_PATH:-"exp/isaid_ori/train"}
GPUS=${GPUS:-1}
PORT=${PORT:-29500}
IMAGE_SIZE=${IMAGE_SIZE:-512}
INTERVAL=${INTERVAL:-1}
RESUME=${RESUME:-""}

# 打印使用说明
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --config CONFIG_PATH       配置文件路径 (默认: configs/isaid_ori.yaml)"
    echo "  --labeled-id-path PATH     标注数据ID列表文件路径 (必需)"
    echo "  --unlabeled-id-path PATH   未标注数据ID列表文件路径 (可选)"
    echo "  --save-path PATH           模型保存路径 (必需)"
    echo "  --gpus NUM                 GPU数量 (默认: 1)"
    echo "  --port PORT                分布式训练端口 (默认: 29500)"
    echo "  --image-size SIZE          输入图像尺寸 (默认: 512)"
    echo "  --interval NUM             验证间隔 (默认: 1)"
    echo "  --resume PATH              预训练模型路径 (可选)"
    echo "  -h, --help                 显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 单卡训练"
    echo "  $0 --config configs/isaid_ori.yaml --labeled-id-path splits/isaid_ori/all/labeled.txt --save-path exp/isaid_ori/train"
    echo ""
    echo "  # 多卡训练 (4卡)"
    echo "  $0 --config configs/isaid_ori.yaml --labeled-id-path splits/isaid_ori/all/labeled.txt --save-path exp/isaid_ori/train --gpus 4"
    echo ""
    echo "  # 从预训练模型恢复训练"
    echo "  $0 --config configs/isaid_ori.yaml --labeled-id-path splits/isaid_ori/all/labeled.txt --save-path exp/isaid_ori/train --resume pretrained/best.pth"
    echo ""
    echo "支持的配置文件:"
    echo "  configs/isaid_ori.yaml, configs/potsdam.yaml, configs/vaihingen.yaml"
    echo "  configs/loveda.yaml, configs/OpenEarthMap.yaml, configs/IRSAMap.yaml, configs/MOTA.yaml"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --labeled-id-path)
            LABELED_ID_PATH="$2"
            shift 2
            ;;
        --unlabeled-id-path)
            UNLABELED_ID_PATH="$2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
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
if [ -z "$LABELED_ID_PATH" ]; then
    echo "错误: 必须指定标注数据列表"
    usage
    exit 1
fi

if [ -z "$SAVE_PATH" ]; then
    echo "错误: 必须指定模型保存路径"
    usage
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 检查标注数据列表是否存在
if [ ! -f "$LABELED_ID_PATH" ]; then
    echo "错误: 标注数据列表不存在: $LABELED_ID_PATH"
    exit 1
fi

# 创建保存目录
mkdir -p "$SAVE_PATH"

# 打印运行信息
echo "========================================"
echo "开始模型训练"
echo "========================================"
echo "配置文件:       $CONFIG"
echo "标注数据列表:   $LABELED_ID_PATH"
echo "未标注数据列表: ${UNLABELED_ID_PATH:-无}"
echo "模型保存路径:   $SAVE_PATH"
echo "GPU数量:        $GPUS"
echo "分布式端口:     $PORT"
echo "图像尺寸:       $IMAGE_SIZE"
echo "验证间隔:       $INTERVAL"
echo "预训练模型:     ${RESUME:-无}"
echo "========================================"

# 构建命令
CMD="python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    1_train_model.py \
    --config \"$CONFIG\" \
    --labeled-id-path \"$LABELED_ID_PATH\" \
    --save-path \"$SAVE_PATH\" \
    --image_size $IMAGE_SIZE \
    --interval $INTERVAL"

# 添加可选参数
if [ -n "$UNLABELED_ID_PATH" ]; then
    CMD="$CMD --unlabeled-id-path \"$UNLABELED_ID_PATH\""
fi

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume \"$RESUME\""
fi

# 运行训练脚本
eval $CMD

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "模型训练完成！"
    echo "========================================"
    echo "模型保存在: $SAVE_PATH"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "模型训练失败！"
    echo "========================================"
    exit 1
fi
