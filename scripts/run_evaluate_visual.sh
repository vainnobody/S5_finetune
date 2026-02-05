#!/bin/bash

# S5_finetune 可视化评估运行脚本
# 用于对模型进行评估并生成可视化结果（类别索引图和RGB可视化图）

# 默认参数
CONFIG=${CONFIG:-"configs/IRSAMap.yaml"}
CKPT_PATH=${CKPT_PATH:-"pretrained/best_dinov3_vit_b_mask_0.5_multi_40k.pth"}
BACKBONE=${BACKBONE:-"dinov3_vit_b"}
INIT_BACKBONE=${INIT_BACKBONE:-"none"}
IMAGE_SIZE=${IMAGE_SIZE:-512}
PRED_DIR=${PRED_DIR:-"exp/visualizations/predictions"}
RGB_DIR=${RGB_DIR:-"exp/visualizations/rgb"}

# 打印使用说明
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --config CONFIG_PATH          配置文件路径 (默认: configs/IRSAMap.yaml)"
    echo "  --ckpt-path CKPT_PATH         模型检查点路径 (默认: pretrained/best_dinov3_vit_b_mask_0.5_multi_40k.pth)"
    echo "  --backbone BACKBONE           骨干网络类型 (默认: dinov3_vit_b)"
    echo "  --init-backbone INIT_BACK     骨干网络初始化方式 (默认: none)"
    echo "  --image-size SIZE             输入图像尺寸 (默认: 512)"
    echo "  --pred-dir PRED_DIR           类别索引预测结果保存目录 (默认: exp/visualizations/predictions)"
    echo "  --rgb-dir RGB_DIR             RGB可视化结果保存目录 (默认: exp/visualizations/rgb)"
    echo "  -h, --help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 使用默认参数运行"
    echo "  $0"
    echo ""
    echo "  # 指定配置文件和检查点"
    echo "  $0 --config configs/isaid_ori.yaml --ckpt-path exp/isaid_ori/vit_b/best.pth"
    echo ""
    echo "  # 指定骨干网络和图像尺寸"
    echo "  $0 --backbone vit_h --image-size 1024"
    echo ""
    echo "  # 自定义输出目录"
    echo "  $0 --pred-dir ./results/preds --rgb-dir ./results/rgb"
    echo ""
    echo "支持的骨干网络:"
    echo "  vit_b, vit_l, vit_h, vit_g"
    echo "  swin_t, swin_b, swin_l"
    echo "  dinov3_vit_b"
    echo "  internimage_xl"
    echo "  R3B_S, vit_b_rvsa, vit_l_rvsa"
    echo ""
    echo "支持的数据集:"
    echo "  IRSAMap, isaid_ori, potsdam, vaihingen, loveda, OpenEarthMap, MOTA"
    echo "  LEVIR, WHU, OSCD, UDD, VDD, UAViD"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --ckpt-path)
            CKPT_PATH="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --init-backbone)
            INIT_BACKBONE="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --pred-dir)
            PRED_DIR="$2"
            shift 2
            ;;
        --rgb-dir)
            RGB_DIR="$2"
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

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 检查检查点文件是否存在
if [ ! -f "$CKPT_PATH" ]; then
    echo "错误: 检查点文件不存在: $CKPT_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$PRED_DIR"
mkdir -p "$RGB_DIR"

# 打印运行信息
echo "========================================"
echo "开始可视化评估"
echo "========================================"
echo "配置文件:     $CONFIG"
echo "检查点:       $CKPT_PATH"
echo "骨干网络:     $BACKBONE"
echo "初始化方式:   $INIT_BACKBONE"
echo "图像尺寸:     $IMAGE_SIZE"
echo "预测保存目录: $PRED_DIR"
echo "RGB保存目录:  $RGB_DIR"
echo "========================================"

# 运行评估脚本
python evaluate_visual.py \
    --config "$CONFIG" \
    --ckpt-path "$CKPT_PATH" \
    --backbone "$BACKBONE" \
    --init_backbone "$INIT_BACKBONE" \
    --image-size "$IMAGE_SIZE" \
    --pred-dir "$PRED_DIR" \
    --rgb-dir "$RGB_DIR"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "可视化评估完成！"
    echo "========================================"
    echo "类别索引图保存在: $PRED_DIR"
    echo "RGB可视化图保存在: $RGB_DIR"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "可视化评估失败！"
    echo "========================================"
    exit 1
fi