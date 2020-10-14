from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, set_stream_logger, set_file_logger
from networks.resnet_big import SupConResNet
from losses import SupConLoss
import logging, datetime
import torch.nn.functional as F

def parse_option():
    parser = argparse.ArgumentParser('argument for training')


    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    parser.add_argument('--sec', action='store_true',
                        help='add sec loss')
    parser.add_argument('--sec_wei', type=float, default=0.0)
    parser.add_argument('--norm_momentum', type=float, default=1.0)
    parser.add_argument('--l2reg', action='store_true',
                        help='add l2reg loss')
    parser.add_argument('--l2reg_wei', type=float, default=0.0)

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--ngpu', type=int, default=2)

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './work_space/{}_models'.format(opt.dataset)
    opt.tb_path = './work_space/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.sec:
        opt.model_name = '{}_sec'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    now_time = datetime.datetime.now().strftime('%m%d_%H%M')
    conf_work_path = '{}_'.format(opt.dataset) + now_time + '_'

    opt.tb_folder = os.path.join(opt.tb_path, conf_work_path + opt.model_name)
    if not os.path.isdir(opt.tb_folder) and opt.local_rank==0:
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, conf_work_path + opt.model_name)
    if not os.path.isdir(opt.save_folder) and opt.local_rank==0:
        os.makedirs(opt.save_folder)

    logging.root.setLevel(logging.INFO)
    set_stream_logger(logging.DEBUG)
    if opt.local_rank==0:
        set_file_logger(work_dir=opt.save_folder, log_level=logging.DEBUG)
        logging.info(f'create {conf_work_path} ...')

    opt.record_norm_mean = None

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=opt.ngpu,
        rank=opt.local_rank,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size // opt.ngpu,
        num_workers=8,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    if opt.ckpt != '':
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        logging.info(f'load model from {opt.ckpt} ...')
        model.load_state_dict(state_dict)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(opt.local_rank))

        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
        )
        criterion = criterion.to(device)
        cudnn.benchmark = True


    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, logger):
    """one epoch training"""
    model.train()

    device = torch.device('cuda:{}'.format(opt.local_rank))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(device, non_blocking=True)
            labels = labels.cuda(device, non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        part_features = model(images)

        tensor_list = [torch.zeros_like(part_features) for i in range(opt.ngpu)]
        torch.distributed.all_gather(tensor_list, part_features)

        if opt.local_rank==0:
            tensor_list[0] = part_features
        elif opt.local_rank==1:
            tensor_list[1] = part_features

        fea00, fea01 = torch.split(tensor_list[0], [bsz, bsz], dim=0)
        fea10, fea11 = torch.split(tensor_list[1], [bsz, bsz], dim=0)

        features = torch.cat([fea00, fea10, fea01, fea11], dim=0)

        assert features.size(1)==128 and features.size(0)==bsz*4

        n_fea = F.normalize(features, dim=1)

        f1, f2 = torch.split(n_fea, [bsz*2, bsz*2], dim=0)
        n_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(n_features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(n_features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        iters_per_epoch = len(train_loader)
        now_iter = (epoch - 1) * iters_per_epoch + idx

        features_norms = torch.norm(features.view(bsz * 4, 128), p=2, dim=1)

        norm_mean = features_norms.mean()
        norm_var = ((features_norms - norm_mean) ** 2).mean()

        # SEC
        if opt.record_norm_mean is not None:
            opt.record_norm_mean = (1 - opt.norm_momentum) * opt.record_norm_mean + opt.norm_momentum * norm_mean.detach()
        else:
            opt.record_norm_mean = norm_mean.detach()

        loss_sec = ((features_norms - opt.record_norm_mean)**2).mean()

        if opt.sec:
            loss = loss + opt.sec_wei * (now_iter) / (opt.epochs * iters_per_epoch) * loss_sec

        loss_l2reg = (features_norms **2).mean()

        if opt.l2reg:
            loss = loss + opt.l2reg_wei * (now_iter) / (opt.epochs * iters_per_epoch) * loss_l2reg

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if opt.local_rank == 0:
            logger.log_value('info/norm_mean', norm_mean.item(), epoch * iters_per_epoch + idx)
            logger.log_value('info/norm_var', norm_var.item(), epoch * iters_per_epoch + idx)
            logger.log_value('info/record_norm_mean', opt.record_norm_mean.item(), epoch * iters_per_epoch + idx)

            logger.log_value('info/loss_sec', loss_sec.item(), epoch * iters_per_epoch + idx)
            logger.log_value('info/loss_l2reg', loss_l2reg.item(), epoch * iters_per_epoch + idx)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            logging.info('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'norm_mean {norm_mean:.3f} (record: {record_norm_mean:.3f}) var {norm_var:.3f}'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, norm_mean=norm_mean.item(),
                   record_norm_mean=opt.record_norm_mean.item(), norm_var=norm_var.item()))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    torch.cuda.set_device(opt.local_rank)

    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=opt.ngpu,
        rank=opt.local_rank,
    )

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    if opt.local_rank == 0:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    else:
        logger = None

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loader.sampler.set_epoch(epoch)
        loss = train(train_loader, model, criterion, optimizer, epoch, opt, logger)
        time2 = time.time()
        logging.info('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        if opt.local_rank == 0:
            logger.log_value('loss', loss, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0 and opt.local_rank == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    if opt.local_rank == 0:
        save_file = os.path.join(
            opt.save_folder, 'last.pth')
        save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
