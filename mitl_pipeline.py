import argparse
import logging
import os
import pprint
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.distributed as dist
import numpy as np
import random
import json
from PIL import Image
from tqdm import tqdm

from dataset.finetune import SemiDataset, ValDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, init_log, intersectionAndUnion, intersectionAndUnionGPU
from util.dist_helper import setup_distributed
from model.semseg.upernet import UperNet
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='MITL Label Refinement Pipeline')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--backbone', type=str, default='vit_b', required=True)
parser.add_argument('--init_backbone', type=str, default='mae', required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--interval', default=1, type=int, help='valid interval')
parser.add_argument('--load', type=str, default='none', choices=['backbone','network', 'none'], help='loaded model part')
parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')

# MITL specific arguments
parser.add_argument('--max-iterations', type=int, default=5, help='max MITL iterations')
parser.add_argument('--train-epochs-per-iter', type=int, default=10, help='training epochs per iteration')
parser.add_argument('--confidence-threshold', type=float, default=0.8, help='confidence threshold for label refinement')
parser.add_argument('--early-stop-threshold', type=float, default=0.01, help='early stop threshold')
parser.add_argument('--save-confidence', action='store_true', help='save confidence maps')


def set_seeds(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def validation_cpu(cfg, args, model, valid_loader):
    """验证函数 - 与 main_finetune.py 风格一致"""
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()

    model.eval()

    for (x, y) in valid_loader:
        x = x.cuda()

        if cfg['eval_mode'] == 'slide_window':
            b, _, h, w = x.shape
            final = torch.zeros(b, cfg['nclass'], h, w).cuda()
            size = cfg['crop_size']
            step = 510
            row, col = 0, 0
            while (row <= int(h / step)):
                col = 0
                while (col <= int(w / step)):
                    sub_input = x[:, :, min(row * step, h - size): min(row * step + size, h),
                                     min(col * step, w - size): min(col * step + size, w)]
                    mask = model(sub_input)
                    if isinstance(mask, dict):
                        mask = mask['out']
                    final[:, :, min(row * step, h - size): min(row * step + size, h),
                          min(col * step, w - size): min(col * step + size, w)] += mask
                    col += 1
                row += 1
            o = final.argmax(dim=1)

        elif cfg['eval_mode'] == 'resize':
            original_shape = x.shape[-2:]
            resized_x = F.interpolate(x, size=cfg['crop_size'], mode='bilinear', align_corners=True)
            resized_o = model(resized_x)
            if isinstance(resized_o, dict):
                resized_o = resized_o['out']
            o = F.interpolate(resized_o, size=original_shape, mode='bilinear', align_corners=True)
            o = o.argmax(dim=1)

        else:
            o = model(x)
            if isinstance(o, dict):
                o = o['out']
            o = o.argmax(dim=1)

        gray = np.uint8(o.cpu().numpy())
        target = np.array(y, dtype=np.int32)
        intersection, union, target_count, predict = intersectionAndUnion(
            gray, target, cfg['nclass'], cfg['ignore_index']
        )

        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target_count).cuda()
        reduced_predict = torch.from_numpy(predict).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)
        dist.all_reduce(reduced_predict)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())
        predict_meter.update(reduced_predict.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10)
    F1_class = 2 * (precise_class * accuracy_class) / (precise_class + accuracy_class)

    if cfg['dataset'] == 'isaid_ori':
        mIoU = np.nanmean(iou_class[1:]) * 100.0
        mAcc = np.nanmean(accuracy_class[1:]) * 100.0
        mF1 = np.nanmean(F1_class[1:]) * 100.0
    else:
        mIoU = np.nanmean(iou_class) * 100.0
        mAcc = np.nanmean(accuracy_class) * 100.0
        mF1 = np.nanmean(F1_class) * 100.0

    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class


@torch.no_grad()
def refine_labels(cfg, args, model, valid_loader, save_path, rank, confidence_threshold):
    """标签优化函数 - 使用滑动窗口推理"""
    model.eval()

    total_changes = 0
    total_pixels = 0

    refined_label_dir = os.path.join(save_path, 'refined_labels')
    if rank == 0:
        os.makedirs(refined_label_dir, exist_ok=True)
        if args.save_confidence:
            os.makedirs(os.path.join(save_path, 'confidence_maps'), exist_ok=True)

    crop_size = cfg['crop_size']

    for idx, (x, y) in enumerate(tqdm(valid_loader, desc='Refining labels', disable=rank != 0)):
        x = x.cuda()
        y = y.cuda()

        b, _, h, w = x.shape

        # 滑动窗口推理获取 logits
        final = torch.zeros(b, cfg['nclass'], h, w).cuda()
        count = torch.zeros(b, 1, h, w).cuda()

        step = crop_size
        row = 0
        while row * step < h:
            col = 0
            while col * step < w:
                h_start = min(row * step, h - crop_size)
                w_start = min(col * step, w - crop_size)
                h_end = min(h_start + crop_size, h)
                w_end = min(w_start + crop_size, w)

                sub_input = x[:, :, h_start:h_end, w_start:w_end]

                # Pad if needed
                if sub_input.shape[-2] < crop_size or sub_input.shape[-1] < crop_size:
                    sub_input = F.pad(sub_input,
                                      (0, crop_size - sub_input.shape[-1],
                                       0, crop_size - sub_input.shape[-2]),
                                      mode='constant', value=0)

                sub_pred = model(sub_input)
                if isinstance(sub_pred, dict):
                    sub_pred = sub_pred['out']

                sub_pred = sub_pred[:, :, :h_end-h_start, :w_end-w_start]

                final[:, :, h_start:h_end, w_start:w_end] += sub_pred
                count[:, :, h_start:h_end, w_start:w_end] += 1

                col += 1
            row += 1

        logits = final / count.clamp(min=1)
        pred = logits.argmax(dim=1)

        # 计算置信度
        probs = F.softmax(logits, dim=1)
        confidence = probs.max(dim=1, keepdim=True)[0]

        # 优化标签: 高置信度区域使用预测，低置信度保留原标签
        refined_label = torch.where(confidence.squeeze(1) > confidence_threshold, pred, y)

        # 统计变化
        valid_mask = (y != cfg['ignore_index'])
        changes = (y != refined_label) & valid_mask
        total_changes += changes.sum().item()
        total_pixels += valid_mask.sum().item()

        # 保存优化后的标签 (仅 rank 0)
        if rank == 0:
            for i in range(b):
                label_np = refined_label[i].cpu().numpy().astype(np.uint8)
                Image.fromarray(label_np).save(
                    os.path.join(refined_label_dir, f'{idx * b + i:06d}.png')
                )

                if args.save_confidence:
                    conf_np = (confidence[i].squeeze().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(conf_np).save(
                        os.path.join(save_path, 'confidence_maps', f'{idx * b + i:06d}.png')
                    )

    # 汇总统计
    total_changes_tensor = torch.tensor(total_changes).cuda()
    total_pixels_tensor = torch.tensor(total_pixels).cuda()
    dist.all_reduce(total_changes_tensor)
    dist.all_reduce(total_pixels_tensor)

    change_rate = total_changes_tensor.item() / max(total_pixels_tensor.item(), 1)

    return {'change_rate': change_rate, 'total_changes': total_changes_tensor.item()}


def main():
    args = parser.parse_args()

    # 兼容 torch.distributed.launch 和 torchrun
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(getattr(args, 'local_rank', 0))

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

        # 保存 MITL 配置
        mitl_config = {
            'max_iterations': args.max_iterations,
            'train_epochs_per_iter': args.train_epochs_per_iter,
            'confidence_threshold': args.confidence_threshold,
            'early_stop_threshold': args.early_stop_threshold,
        }
        with open(os.path.join(args.save_path, 'mitl_config.json'), 'w') as f:
            json.dump(mitl_config, f, indent=2)

    cudnn.enabled = True
    cudnn.benchmark = True

    # 构建模型
    model = UperNet(args, cfg)

    lr = {"vit_h": 0.00005, "vit_l": 0.00005, "vit_b": 0.00005,
          "dinov3_vit_b": 0.00005, "dinov3_vit_b_mae": 0.00005,
          "vit_l_rvsa": 0.00005}.get(args.backbone, 0.0001)

    # 加载预训练权重
    if args.load == 'network' and args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            ckpt_dict = checkpoint.get('model', checkpoint)

            if list(ckpt_dict.keys())[0].startswith('module.'):
                ckpt_dict = {k[7:]: v for k, v in ckpt_dict.items()}

            model_dict = model.state_dict()
            filtered_ckpt_dict = {}
            for k, v in ckpt_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_ckpt_dict[k] = v
                elif rank == 0:
                    logger.warning(f"Skipping parameter: {k}")

            model_dict.update(filtered_ckpt_dict)
            model.load_state_dict(model_dict, strict=False)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    # 优化器
    from mmengine.optim import build_optim_wrapper
    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.05),
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9)
    )
    optimizer = build_optim_wrapper(model, optim_wrapper)

    # DDP 设置
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False,
        output_device=local_rank, find_unused_parameters=True
    )

    if args.backbone in ['vit_l', 'vit_b', 'vit_h', 'vit_l_rvsa']:
        model._set_static_graph()

    # 损失函数
    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    # 数据集
    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', size=cfg['crop_size'], id_path=args.labeled_id_path)
    valset = ValDataset(cfg['dataset'], cfg['data_root'], 'val', id_path=args.labeled_id_path)

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=(trainsampler is None),
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)

    valsampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=world_size, rank=rank)
    val_batch = 1 if cfg['dataset'] in ['OpenEarthMap', 'MOTA'] else 8
    valloader = DataLoader(valset, batch_size=val_batch, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    scaler = torch.cuda.amp.GradScaler()
    amp = cfg.get('amp', True)

    # 计算总 epoch 数
    total_epochs = args.max_iterations * args.train_epochs_per_iter
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optimizer, total_epochs, eta_min=0, last_epoch=-1
    )

    # MITL 迭代
    previous_best = 0.0
    global_epoch = 0
    history = []

    for mitl_iter in range(args.max_iterations):
        if rank == 0:
            logger.info('=' * 50)
            logger.info('MITL Iteration {}/{}'.format(mitl_iter + 1, args.max_iterations))
            logger.info('=' * 50)

        iter_save_path = os.path.join(args.save_path, f'iteration_{mitl_iter + 1}')
        if rank == 0:
            os.makedirs(iter_save_path, exist_ok=True)

        # 训练阶段
        for epoch in range(args.train_epochs_per_iter):
            if rank == 0:
                logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                    global_epoch + 1, optimizer.param_groups[0]['lr'], previous_best))

            total_loss = AverageMeter()
            trainsampler.set_epoch(global_epoch)
            for i, (img, mask) in enumerate(trainloader):
                img, mask = img.cuda(), mask.cuda()

                with torch.cuda.amp.autocast(enabled=amp):
                    model.train()
                    pred = model(img)
                    if isinstance(pred, dict):
                        pred = pred['out']
                    loss = criterion(pred, mask)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss.update(loss.item())

                iters = global_epoch * len(trainloader) + i

                if rank == 0:
                    writer.add_scalar('train/loss', loss.item(), iters)

                if (i % max(2, len(trainloader) // 8) == 0) and rank == 0:
                    logger.info('Iters: {:}, Loss: {:.3f}'.format(i, loss.item()))

            scheduler.step()
            global_epoch += 1

        # 验证阶段
        start_time = time.time()
        mIoU, mAcc, mF1, allAcc, iou_class, F1_class = validation_cpu(cfg, args, model, valloader)
        end_time = time.time()

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.4f}'.format(
                    cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))

            logger.info('Validation [{}/{}]: mIoU/mAcc/mF1/allAcc {:.2f}/{:.2f}/{:.2f}/{:.2f}. Cost {:.2f} secs'.format(
                mitl_iter + 1, args.max_iterations, mIoU, mAcc, mF1, allAcc, end_time - start_time))

            writer.add_scalar('eval/mIoU', mIoU, mitl_iter)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, mitl_iter)

        # 标签优化阶段
        if rank == 0:
            logger.info('Refining labels...')
        refine_stats = refine_labels(
            cfg, args, model, valloader, iter_save_path, rank, args.confidence_threshold
        )

        if rank == 0:
            logger.info('Refinement change rate: {:.4f}'.format(refine_stats['change_rate']))

        # 记录历史
        iter_record = {
            'iteration': mitl_iter + 1,
            'mIoU': mIoU,
            'mAcc': mAcc,
            'mF1': mF1,
            'allAcc': allAcc,
            'change_rate': refine_stats['change_rate']
        }
        history.append(iter_record)

        # 保存 checkpoint
        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': global_epoch,
                'mitl_iteration': mitl_iter + 1,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
                logger.info('New best mIoU: {:.2f}'.format(previous_best))

        # 早停检查
        if refine_stats['change_rate'] < args.early_stop_threshold:
            if rank == 0:
                logger.info('Early stopping: change rate {:.4f} < {:.4f}'.format(
                    refine_stats['change_rate'], args.early_stop_threshold))
            break

        # 更新训练数据集使用优化后的标签
        refined_split_file = os.path.join(iter_save_path, 'refined_split.txt')
        if rank == 0:
            # 创建新的 split 文件
            with open(args.labeled_id_path, 'r') as f:
                original_lines = f.readlines()
            refined_label_dir = os.path.join(iter_save_path, 'refined_labels')

            with open(refined_split_file, 'w') as f:
                for i, line in enumerate(original_lines):
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        refined_label_path = os.path.join(refined_label_dir, f'{i:06d}.png')
                        f.write(f'{img_path} {refined_label_path}\n')

        dist.barrier()  # 等待 rank 0 完成 split 文件写入

        # 重新创建数据集
        trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', size=cfg['crop_size'], id_path=refined_split_file)
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
        trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=(trainsampler is None),
                                 pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)

    # 保存最终结果
    if rank == 0:
        logger.info('=' * 50)
        logger.info('MITL Pipeline Completed')
        logger.info('Best mIoU: {:.2f}'.format(previous_best))
        logger.info('Total iterations: {}'.format(len(history)))

        with open(os.path.join(args.save_path, 'mitl_history.json'), 'w') as f:
            json.dump(history, f, indent=2)


if __name__ == '__main__':
    set_seeds(1234)
    main()
