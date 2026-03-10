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
# from evaluate import evaluate
from dataset.finetune import SemiDataset, ValDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, init_log, intersectionAndUnion, intersectionAndUnionGPU
from util.dist_helper import setup_distributed
from model.semseg.dpt import DPT
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', '--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--interval', default=1, type=int, help='valid interval')
parser.add_argument('--resume', type=str, default='/data1/users/lvliang/project_123/S5_finetune/pretrained/best_dinov3_vit_b_mask_0.5_multi_40k.pth', help='resume name')



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

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()

    model.eval()

    for (x, y) in valid_loader:
        x = x.cuda()

        if cfg['eval_mode'] == 'slide_window':
            b, _, h, w = x.shape    # 获取输入图像的尺寸 (batch, channels, height, width)
            final = torch.zeros(b, cfg['nclass'], h, w).cuda()  # 用于存储最终预测结果
            size = cfg['crop_size']
            step = 510
            b = 0
            a = 0
            while (a <= int(h / step)):
                while (b <= int(w / step)):
                    sub_input = x[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)]
                    # print("sub_input.shape", sub_input.shape)
                    mask = model(sub_input) 
                    final[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)] += mask
                    b += 1
                b = 0
                a += 1
            o = final.argmax(dim=1)
        
        elif cfg['eval_mode'] == 'resize':
        # 使用缩放方式进行预测
            original_shape = x.shape[-2:]  # 保存原始图像的尺寸 (h, w)
            resized_x = F.interpolate(x, size=cfg['crop_size'], mode='bilinear', align_corners=True)
            resized_o = model(resized_x)   
            # 将预测结果复原到原始尺寸
            o = F.interpolate(resized_o, size=original_shape, mode='bilinear', align_corners=True)
            o = o.argmax(dim=1)

        else:
            # 直接进行预测（非滑动窗口模式）
            o = model(x)
            o = o.max(1)[1]
        gray = np.uint8(o.cpu().numpy())
        target = np.array(y, dtype=np.int32)
        intersection, union, target, predict = intersectionAndUnion(gray, target, cfg['nclass'], cfg['ignore_index'])
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()
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
    F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class)
    if cfg['dataset'] == 'isaid_ori':
        mIoU = np.nanmean(iou_class[1:]) * 100.0
        mAcc = np.nanmean(accuracy_class[1:]) * 100.0
        mF1 = np.nanmean(F1_class[1:]) * 100.0
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    else:
        mIoU = np.nanmean(iou_class) * 100.0
        mAcc = np.nanmean(accuracy_class) * 100.0
        mF1 = np.nanmean(F1_class) * 100.0
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class



def generate_masks(x_tokens: torch.Tensor, mask_ratio: float):
    """
    x_tokens: (B, L, D) 仅用于拿到 B, L（也可以不用真实 token，只要知道 L 就行）
    return:
      masks_bool: (B, L)  True=mask, False=keep
      ids_keep:   (B, len_keep)
    """
    B, L, D = x_tokens.shape
    device = x_tokens.device
    len_keep = int(L * (1 - mask_ratio))

    # ✅ 整个 batch 共用同一个随机排列
    perm = torch.randperm(L, device=device)
    ids_keep_single = perm[:len_keep]
    ids_keep_single, _ = torch.sort(ids_keep_single)  # 可选：保持原始空间顺序

    # 扩展到整个 batch
    ids_keep = ids_keep_single.unsqueeze(0).expand(B, -1)  # (B, len_keep)

    masks_bool = torch.ones(B, L, device=device, dtype=torch.bool)  # 全 mask
    masks_bool.scatter_(1, ids_keep, False)  # keep 的位置设为 False

    return masks_bool, ids_keep

def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        "small": {
            "encoder_size": "small",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "base": {
            "encoder_size": "base",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "large": {
            "encoder_size": "large",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "giant": {
            "encoder_size": "giant",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }
    backbone_size = cfg["backbone"].split("_")[-1]
    backbone_version = cfg["backbone"].split("_")[0]
    model = DPT(
            **{**model_configs[backbone_size], "nclass": cfg["nclass"]},
            backbone_version=backbone_version,
        )


    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    optimizer = AdamW(
            [
                {
                    "params": [p for p in model.backbone.parameters() if p.requires_grad],
                    "lr": cfg["lr"],
                },
                {
                    "params": [
                        param
                        for name, param in model.named_parameters()
                        if "backbone" not in name
                    ],
                    "lr": cfg["lr"] * cfg["lr_multi"],
                },
            ],
            lr=cfg["lr"],
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

    # if args.backbone == 'vit_l' or args.backbone == 'vit_b' or args.backbone == 'vit_h' or args.backbone == 'vit_l_rvsa':
    #     model._set_static_graph()

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', size=cfg['crop_size'], id_path=args.labeled_id_path)
    valset = ValDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size,rank=rank)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=(trainsampler is None),
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=world_size, rank=rank)
    
    val_batch = 1 if cfg['dataset'] == 'OpenEarthMap' else 8

    valloader = DataLoader(valset, batch_size=val_batch, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    scaler = torch.cuda.amp.GradScaler()
    amp = cfg['amp']

    # if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    #     checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     epoch = checkpoint['epoch']
    #     previous_best = checkpoint['previous_best']
        
        # if rank == 0:
        #     logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        
        total_loss = AverageMeter()
        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            with torch.cuda.amp.autocast(enabled=amp):
                model.train()
                pred = model(img)
                sup_loss = criterion(pred, mask)
                torch.distributed.barrier()
                optimizer.zero_grad()
                loss = scaler.scale(sup_loss)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss.update(sup_loss)
            iters = epoch * len(trainloader) + i

            if rank == 0:
                writer.add_scalar('train/loss_all', sup_loss.item(), iters)
                writer.add_scalar('train/loss_x', sup_loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                # logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, sup_loss.item()))

        scheduler.step()

        if (epoch + 1) % args.interval == 0:
            start_time = time.time()
            mIoU, mAcc, mF1, allAcc, iou_class, F1_class = validation_cpu(cfg, args, model, valloader)
            end_time = time.time()

            if rank == 0:
                for (cls_idx, iou) in enumerate(iou_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'IoU: {:.4f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                # logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                logger.info('Last: validation epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}. Cost {:.4f} secs \n'.format(epoch+1, cfg['epochs'], mIoU, mAcc, mF1, allAcc, end_time-start_time))

                for (cls_idx, F1) in enumerate(F1_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'F1 score: {:.4f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], F1))
                # logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                logger.info('Last: validation epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}. Cost {:.4f} secs'.format(epoch+1, cfg['epochs'], mIoU, mAcc, mF1, allAcc, end_time-start_time))
                    
                writer.add_scalar('eval/mIoU', mIoU, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

            is_best = mIoU > previous_best
            previous_best = max(mIoU, previous_best)
            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'previous_best': previous_best,
                }
                torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
                if is_best:
                    torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    set_seeds(1234)
    main()
