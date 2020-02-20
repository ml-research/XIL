import numpy as np
import torch

torch.manual_seed(17)
np.random.seed(17)

import argparse
import os
import random
import shutil
import time
import warnings
import sys

from tensorboardX import SummaryWriter
from Plant_Phenotyping.dataset_hs import LeafDataset as DeepDataset

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from Plant_Phenotyping.network_hs import ConvNetDefault as NetDefault

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from Plant_Phenotyping.rrr_loss_hs import rrr_loss_function
from utils import PytorchProcessName

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4), (rule of thump: (2 to 4) * numgpus)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--correct-rwr', default='', type=str, metavar='PATH',
                    help='path to rwr model (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-gc', '--gradcam', dest='gradcam', action='store_true',
                    help='gradcam of model on validation and train set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=17, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpus', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--selected_activation', default='relu6', type=str,
                    help='one of the standard activation functions or a pade activation function')
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--data_path', type=str, default='/')
parser.add_argument('--arch', type=str, default='default')
parser.add_argument('--mask', default=0, type=int,
                    help='')
parser.add_argument('--mask_eval', default=0, type=int,
                    help='')
parser.add_argument('--l2_grads', default=8, type=int,
                    help='')
parser.add_argument('--cv_splits', default=1, type=int,
                    help='')
parser.add_argument('--cv_current_split', default=0, type=int,
                    help='')
parser.add_argument('--weighted-rrr', action="store_true")

best_acc1 = 0

writer = None


def totimestring(seconds):
    day = seconds // (24 * 3600)
    time = seconds % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    return "{}d:{}h:{}m".format(int(day), int(hour), int(minutes))


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpus is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # Simply call main_worker function
    main_worker(list(map(int, args.gpus.strip().split(','))), args)


def main_worker(gpu, args):
    global best_acc1
    global writer

    args.gpus = gpu
    torch.set_num_threads(len(gpu) * 10)
    num_workers = len(gpu) * args.workers
    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    args.save_path = os.path.join(args.save_path, args.arch)

    if args.arch == 'default':
        model = NetDefault(elu=False, avgpool=False).cuda()
    else:
        raise ValueError("dont use that")

    print("GPUs:", args.gpus)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpus is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1  # .cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()
    elif args.correct_rwr:
        if os.path.isfile(args.correct_rwr):
            print("=> loading checkpoint of right for the wrong reasons model '{}'".format(args.correct_rwr))
            checkpoint = torch.load(args.correct_rwr)
            args.start_epoch = 100#checkpoint['epoch']
            #best_acc1 = checkpoint['best_acc1']
            if args.gpus is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1  # .cuda()
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
                  .format(args.correct_rwr, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.correct_rwr))
            exit()
    print("Start loading data")
    train_dataset = DeepDataset(data_path=os.path.join(args.data_path),
                                patch_length=213,
                                patch=True,
                                single_image=True,
                                use_labels=None,
                                preload_center_crop=0,
                                n_shots=-1,
                                use_negative=0,
                                binary_dai=True, fiveD=True,
                                step_wavelength=5, clip_wavelength=64,
                                mask_background=args.mask,
                                split=args.cv_current_split,
                                cv_n_splits=args.cv_splits,
                                )  # mask_background: 0 = dont use mask, 1= use mask on training data, 2= return mask with x

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    val_dataset = DeepDataset(data_path=os.path.join(args.data_path),
                              patch_length=213,
                              patch=True,
                              single_image=True,
                              use_labels=None,
                              preload_center_crop=0,
                              n_shots=-1,
                              use_negative=0,
                              mode='eval', binary_dai=True, fiveD=True,
                              step_wavelength=5, clip_wavelength=64,
                              mask_background=args.mask_eval,
                              split=args.cv_current_split,
                              cv_n_splits=args.cv_splits,
                              )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)

    # define loss function (criterion) and optimizer
    if train_dataset.mask_background == 2:
        criterion = nn.NLLLoss(weight=train_dataset.class_balance).cuda()
    else:
        criterion = nn.CrossEntropyLoss(weight=train_dataset.class_balance).cuda()

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args)

    if args.gradcam:
        # grad_cam(train_loader, model, args, "train") comment in for gradcams generated by training data
        grad_cam(val_loader, model, args, "eval")

    if args.evaluate or args.gradcam:
        # exit
        return

    def save_target_output(self, input, output):
        model.target_output = output

    def forward_pass_on_convolutions(x, target_layer):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        for module_pos, module in x.features._modules.items():
            # print("Module ",module, " at position ", module_pos)
            if int(module_pos) == target_layer:
                module.register_forward_hook(save_target_output)

    forward_pass_on_convolutions(model, target_layer=4)

    writer = SummaryWriter(args.save_path)

    # log learning configuration
    writer.add_scalar('configuration/weight_decay', args.weight_decay)
    writer.add_scalar('configuration/momentum', args.momentum)
    writer.add_scalar('configuration/batch_size', args.batch_size)
    # writer.add_scalar('configuration/network', args.arch)
    writer.add_scalar('configuration/seed', args.seed)

    # set process name
    procname = PytorchProcessName(args.epochs, name="ML Deepplant %s" % ('hsi classification'))

    print(procname.name)
    print("---" * 42)

    procname.start()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, [optimizer], [], epoch, args)

        # evaluate on validation set
        acc1, val_loss = validate(val_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        procname.update_epoch(epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, epoch, is_best, args=args)

    ##
    writer.close()


def train(train_loader, model, criterion, optimizers, schedulers, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_rrr = AverageMeter('Loss Right Reason', ':.4e')
    losses_ra = AverageMeter('Loss Right Answer', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_balanced = AverageBalancedAccuracyMeter('BalancedAcc@1', ':6.2f')

    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, losses_rrr, losses_ra,
                             top1, top1_balanced,
                             prefix="Epoch: [{}]".format(epoch))

    n_batches = len(train_loader)
    print("Num Batches per epoch: ", n_batches)
    # switch to train mode
    model.train()

    end = time.time()

    # log learning rates
    for i, optimizer in enumerate(optimizers):
        writer.add_scalar('lr/{}'.format(i + 1), optimizer.param_groups[0]["lr"], epoch + 1)

    A = None

    class_balance_rrr = None
    if args.weighted_rrr:
        class_balance_rrr = train_loader.dataset.class_balance

    for batch_idx, (x, y, _) in enumerate(train_loader):
        target = y[0]
        if train_loader.dataset.mask_background == 2:
            input = x[0]
            A = x[1].float().cuda()
        else:
            input = x
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpus is not None:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        if train_loader.dataset.mask_background == 2:
            loss, loss_ra, loss_rrr = rrr_loss_function(A=A, X=model.target_output, y=target, logits=output,
                                                        criterion=criterion,
                                                        class_weights=class_balance_rrr,
                                                        l2_grads=args.l2_grads)
        else:
            loss = criterion(output, target)
            loss_rrr = torch.Tensor([0])
            loss_ra = torch.Tensor([0])

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        losses_rrr.update(loss_rrr.item(), input.size(0))
        losses_ra.update(loss_ra.item(), input.size(0))
        top1.update(acc1[0].detach().cpu().numpy().item(), input.size(0))
        batch_pred, batch_target = getPredAndTarget(output, target)
        top1_balanced.update(batch_pred, batch_target)
        # compute gradient and do SGD step
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()

        # clip_grad_norm_(model.parameters(), 5., norm_type=2)

        # update step
        for optimizer in optimizers:
            optimizer.step()

        for idx_scheduler, scheduler in enumerate(schedulers):
            if batch_idx == n_batches - 1:
                scheduler["scheduler"].step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.print(batch_idx)

    # log to tensorboard
    writer.add_scalar('train/loss', losses.avg, epoch + 1)
    writer.add_scalar('train/loss_rrr', losses_rrr.avg, epoch + 1)
    writer.add_scalar('train/loss_ra', losses_ra.avg, epoch + 1)
    writer.add_scalar('train/accuracy@1', top1.avg, epoch + 1)
    writer.add_scalar('train/accuracy@1_balanced', top1_balanced.avg, epoch + 1)


def validate(val_loader, model, criterion, epoch, args):
    log_softmax = torch.nn.LogSoftmax(dim=1).cuda()
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_balanced = AverageBalancedAccuracyMeter('BalancedAcc@1', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top1_balanced,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    y_pred = []
    y_true = []

    if args.evaluate:
        stdout = sys.stdout
        sys.stdout = open(os.path.join(os.path.dirname(args.resume), 'evaluation{}.txt'.format(os.path.basename(args.resume).replace("checkpoint", "").replace(".pth.tar", ""))), 'w')
    with torch.no_grad():
        end = time.time()
        for i, (input, y, _) in enumerate(val_loader):
            if val_loader.dataset.mask_background == 2:
                input = input[0]
            target = y[0]
            sample_real_target = y[3][3]
            sample_id = y[3][-1]
            # val_loader.dataset.showSamples((input, y))

            if args.gpus is not None:
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            output = log_softmax(output)
            loss = criterion(output, target)

            batch_pred, batch_target = getPredAndTarget(output, target)

            pred = batch_pred.copy()
            # change pred to real if it correct
            for p_idx, p in enumerate(pred):
                real_target = sample_real_target[p_idx].detach().cpu().numpy().item()
                if p == 1 and real_target != 0:
                    pred[p_idx] = real_target
                else:
                    pass

            y_pred += pred
            y_true += sample_real_target.detach().cpu().numpy().tolist()

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))

            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0].detach().cpu().numpy().item(), input.size(0))
            top1_balanced.update(batch_pred, batch_target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
        print("-- "*42)
        print(' *Total Acc@1 {top1.avg:.2f} | Acc@1 Balanced {top1_b.avg:.2f} '
              .format(top1=top1, top1_b=top1_balanced))

        if args.evaluate:
            cm_cnt = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=np.unique(y_true))
            cm_acc = cm_cnt.copy().astype("float")
            for c_idx, c in enumerate(cm_acc):
                cm_acc[c_idx] = [e / np.sum(c) for e in c]

            import pandas as pd

            df = pd.DataFrame(cm_cnt)
            print(df)
            print("-- " * 42)
            df = pd.DataFrame(cm_acc)
            print(df)
            print("-- " * 42)
            print("-- " * 42)
            sys.stdout = stdout
            # print(summary(model, (1, 64, 213, 213)))

    if writer is not None:
        writer.add_scalar('validate/loss', losses.avg, epoch + 1)
        writer.add_scalar('validate/accuracy@1', top1.avg, epoch + 1)
        writer.add_scalar('validate/accuracy@1_balanced', top1_balanced.avg, epoch + 1)

    return top1_balanced.avg, losses.avg


def grad_cam(data_loader, model, args, save_path_suffix):
    from Plant_Phenotyping.hs_utils.gradcam import GradCam, save_class_activation_images3D
    from PIL import Image
    import pickle
    # switch to evaluate mode
    model.eval()
    save_dir = os.path.join(os.path.dirname(args.resume), 'gradcam_' + str(os.path.basename(args.resume).split(".")[0]))
    save_dir = os.path.join(save_dir, save_path_suffix)
    os.makedirs(save_dir, exist_ok=True)
    print("Num batches", len(data_loader))
    glob_min = 10000
    glob_max = -10000
    for i, (input, y, _) in enumerate(tqdm(data_loader)):
        target = y[0]
        sample_real_target = y[3][3]
        sample_id = y[3][-1]
        if args.gpus is not None:
            input = input.cuda()
            target = target.cuda()

        for idx_x, x in enumerate(input):
            # compute output
            with torch.no_grad():
                output = model(x.unsqueeze(0))
                _, y_pred = torch.max(output.data, 1)
                y_pred = y_pred.detach().cpu().numpy().item()
            save_filename = os.path.join(save_dir,
                                         '{}_class_{}_y_{}_ypred_{}.npy'.format(
                                             # sample_real_target[idx_x],
                                             sample_id[idx_x].replace(",", "_"),
                                             y_pred,
                                             target[idx_x],
                                             y_pred))

            #if os.path.isfile(save_filename):
            #    continue
            grad_cam_res = GradCam(model, target_layer=4)

            orig_img = data_loader.dataset.get_samples(([x], [None]))[0]
            orig_img = Image.fromarray(orig_img)

            cam, num_channels_upscaling, spectral_cam = grad_cam_res.generate_cam_3D(x.unsqueeze(0), [y_pred])
            cam = cam[::num_channels_upscaling, :, :]
            glob_min = np.minimum(np.min(cam), glob_min)
            glob_max = np.maximum(np.max(cam), glob_max)

            cam_for_spray = cam.copy()
            cam_for_spray = np.mean(cam_for_spray, axis=0)

            dict_dump = {"cams": cam,
                         "spatial_cams": cam_for_spray,
                         "spectral_cam": spectral_cam,
                         'samples': data_loader.dataset.sample_wavelengths}

            #pickle.dump(dict_dump, open(save_filename, "wb"))
            """save_class_activation_images3D(orig_img, cam,
                                           os.path.join(save_dir,
                                                        'dai{}_id{}_class{}_gt{}_pred{}.jpg'.format(
                                                            sample_real_target[idx_x],
                                                            sample_id[idx_x],
                                                            y_pred,
                                                            target[idx_x],
                                                            y_pred)),
                                           samples=data_loader.dataset.sample_wavelengths,
                                           spectral_activations=spectral_cam,
                                           orig_wavelength=data_loader.dataset.wavelength_vnir)"""
            save_class_activation_images3D(orig_img, cam,
                                           os.path.join(os.path.join(save_dir, "plots"),
                                                        'dai{}_id{}_class{}_gt{}_pred{}.jpg'.format(
                                                            sample_real_target[idx_x],
                                                            sample_id[idx_x],
                                                            y_pred,
                                                            target[idx_x],
                                                            y_pred)),
                                           samples=data_loader.dataset.sample_wavelengths,
                                           spectral_activations=spectral_cam,
                                           orig_wavelength=data_loader.dataset.wavelength_vnir, wv_plot=False)


def save_checkpoint(state, epoch, is_best, args, filename='checkpoint.pth.tar'):
    save_path_checkpoint = os.path.join(args.save_path, filename)
    save_path_checkpoint_best = os.path.join(args.save_path, 'model_best.pth.tar')
    torch.save(state, save_path_checkpoint)
    if epoch % 20 == 0 and epoch != 0:
        torch.save(state, save_path_checkpoint.replace('checkpoint.pth.tar', 'checkpoint{}.pth.tar'.format(epoch)))
    if is_best:
        shutil.copyfile(save_path_checkpoint, save_path_checkpoint_best)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageBalancedAccuracyMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.pred = []
        self.target = []
        self.avg = 0

    def update(self, pred, target):
        self.pred += pred
        self.target += target
        self.avg = balanced_accuracy(self.target, self.pred) * 100

    def __str__(self):
        fmtstr = '{name} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def getPredAndTarget(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        return pred.view(-1).detach().cpu().numpy().tolist(), target.view(-1).detach().cpu().numpy().tolist()


def balanced_accuracy(target, pred):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    res = balanced_accuracy_score(target, pred)
    return res


if __name__ == '__main__':
    main()
