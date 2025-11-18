import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather
import torch.utils.data.distributed
import SimpleITK as sitk
from MILoss import MILoss, smooth_loss
from losses import NCC
from monai.losses import DiceCELoss, DiceLoss


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)  
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss) 
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class AverageMeter(object):

    def __init__(self):
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
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def train_epoch(model,
                loader,
                optimizer,
                scheduler,
                epoch,
                args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    #### loss function ####
    mse = nn.MSELoss()
    wmse = nn.MSELoss(reduction='none')
    smoothloss = smooth_loss(penalty='l2', loss_mult=2)
    loss_func = nn.CrossEntropyLoss(reduction='none')
    ce = nn.CrossEntropyLoss()
    ncc = NCC().loss
    diceloss = DiceLoss()
    focalloss = FocalLoss()
    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6
    )
    step_num = 0
    for idx, batch_data in enumerate(loader):
        data, label, weighted = batch_data['image'], batch_data['label'], batch_data['weight']
        print(data.shape[0], data.shape[1],data.shape[2],data.shape[3],data.shape[4])
        print(label.shape[0], label.shape[1], label.shape[2], label.shape[3], label.shape[4])

        #data = torch.tensor(data)
        #label = torch.tensor(label)
        #weighted = torch.tensor(weighted)


        data = data/10000.0

        for n_t in range(3):
            for n_tt in range(100):
                in_data = np.zeros([3,1,64,64,64])
                in_label = np.zeros([3,1,64,64,64])
                in_weight = np.zeros([3,1,64,64,64])

                in_data[0,:,:,:,:]=data[n_t*300+n_tt,:,:,:,:]
                in_label[0, :, :, :, :] = label[n_t * 300 + n_tt, :, :, :, :]
                in_weight[0, :, :, :, :] = weighted[n_t * 300 + n_tt, :, :, :, :]

                in_data[1, :, :, :, :] = data[n_t * 300 + 100 + n_tt, :, :, :, :]
                in_label[1, :, :, :, :] = label[n_t * 300 + 100 + n_tt, :, :, :, :]
                in_weight[1, :, :, :, :] = weighted[n_t * 300 + 100 + n_tt, :, :, :, :]

                in_data[2, :, :, :, :] = data[n_t * 300 + 200 + n_tt, :, :, :, :]
                in_label[2, :, :, :, :] = label[n_t * 300 + 200 + n_tt, :, :, :, :]
                in_weight[2, :, :, :, :] = weighted[n_t * 300 + 200 + n_tt, :, :, :, :]

                if n_tt==99:
                    data_save = in_data[0, 0, :, :, :]
                    s_path = './img' + str(0) + '.hdr'
                    out = sitk.GetImageFromArray(data_save)
                    sitk.WriteImage(out, s_path)

                    label_save = in_label[0, 0, :, :, :]
                    s_path = './label' + str(0) + '.hdr'
                    out = sitk.GetImageFromArray(label_save)
                    sitk.WriteImage(out, s_path)

                    weighted_save = in_weight[0, 0, :, :, :]
                    s_path = './weighted' + str(0) + '.hdr'
                    out = sitk.GetImageFromArray(weighted_save)
                    sitk.WriteImage(out, s_path)

                    data_save = in_data[1, 0, :, :, :]
                    s_path = './img' + str(1) + '.hdr'
                    out = sitk.GetImageFromArray(data_save)
                    sitk.WriteImage(out, s_path)

                    label_save = in_label[1, 0, :, :, :]
                    s_path = './label' + str(1) + '.hdr'
                    out = sitk.GetImageFromArray(label_save)
                    sitk.WriteImage(out, s_path)

                    weighted_save = in_weight[1, 0, :, :, :]
                    s_path = './weighted' + str(1) + '.hdr'
                    out = sitk.GetImageFromArray(weighted_save)
                    sitk.WriteImage(out, s_path)

                    data_save = in_data[2, 0, :, :, :]
                    s_path = './img' + str(2) + '.hdr'
                    out = sitk.GetImageFromArray(data_save)
                    sitk.WriteImage(out, s_path)

                    label_save = in_label[2, 0, :, :, :]
                    s_path = './label' + str(2) + '.hdr'
                    out = sitk.GetImageFromArray(label_save)
                    sitk.WriteImage(out, s_path)

                    weighted_save = in_weight[2, 0, :, :, :]
                    s_path = './weighted' + str(2) + '.hdr'
                    out = sitk.GetImageFromArray(weighted_save)
                    sitk.WriteImage(out, s_path)

                in_data = torch.tensor(in_data)
                in_label = torch.tensor(in_label)
                in_weight = torch.tensor(in_weight)
                in_data, in_label, in_weight = in_data.cuda(args.rank), in_label.cuda(args.rank), in_weight.cuda(args.rank)

                for param in model.parameters(): param.grad = None
                logits = model(in_data.float())

                loss_matrix = loss_func(logits, in_label.squeeze().long())
                loss = torch.mean(torch.mul(loss_matrix, in_weight))

                print('#loss:', loss)
                logits = logits.detach().cpu().numpy()
                pre = np.argmax(logits, 1)
                print(pre.shape[0], pre.shape[1], pre.shape[2], pre.shape[3])

                if n_tt==99:
                    data_save = in_data[0, 0, :, :, :].cpu().numpy()
                    s_path = './image1.hdr'
                    out = sitk.GetImageFromArray(data_save)
                    sitk.WriteImage(out, s_path)

                    s_path = './pre1.hdr'
                    out = sitk.GetImageFromArray(logits[0, 1, :, :, :])
                    sitk.WriteImage(out, s_path)

                    data_save = in_data[1, 0, :, :, :].cpu().numpy()
                    s_path = './image2.hdr'
                    out = sitk.GetImageFromArray(data_save)
                    sitk.WriteImage(out, s_path)

                    s_path = './pre2.hdr'
                    out = sitk.GetImageFromArray(logits[1, 1, :, :, :])
                    sitk.WriteImage(out, s_path)

                    data_save = in_data[2, 0, :, :, :].cpu().numpy()
                    s_path = './image3.hdr'
                    out = sitk.GetImageFromArray(data_save)
                    sitk.WriteImage(out, s_path)

                    s_path = './pre3.hdr'
                    out = sitk.GetImageFromArray(logits[2, 1, :, :, :])
                    sitk.WriteImage(out, s_path)

                loss.backward()
                optimizer.step()

                run_loss.update(loss.item(), n=args.batch_size)

                print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                      'loss: {:.4f}'.format(run_loss.avg),
                      'time {:.2f}s'.format(time.time() - start_time))

                start_time = time.time()
                for param in model.parameters(): param.grad = None
                step_num = step_num + 1
                modelname = 'step-model-all.pt'

                if (step_num + 1) % 1000 == 0:
                    save_checkpoint(model, (epoch), args, filename=modelname,
                                    best_acc=0,
                                    optimizer=optimizer,
                                    scheduler=scheduler)

    return run_loss.avg


def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict()
    save_dict = {
        'epoch': epoch,
        'best_acc': best_acc,
        'state_dict': state_dict
    }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)


def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 acc_func,
                 args,
                 scheduler=None,
                 start_epoch=0,
                 post_label=None,
                 post_pred=None):
    writer = None

    writer = SummaryWriter(log_dir=args.logdir)
    print('Writing Tensorboard logs to ', args.logdir)
    scaler = None

    val_acc_max = 0.
    for epoch in range(start_epoch, args.max_epochs):
        print(time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()
        train_loss = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 scheduler,
                                 epoch=epoch,
                                 args=args)

        print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
              'time {:.2f}s'.format(time.time() - epoch_time), 'lr: {:,.4f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        if args.rank == 0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False

        modelname = 'epoch' + str(epoch + 0) + 'model-all.pt'

        if (epoch + 1) % 1 == 0:
            save_checkpoint(model, (epoch), args, filename=modelname,
                            best_acc=0,
                            optimizer=optimizer,
                            scheduler=scheduler)
        '''
        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            val_avg_acc = val_epoch(model,
                                    val_loader,
                                    epoch=epoch,
                                    acc_func=dice,
                                    args=args,
                                    post_label=post_label,
                                    post_pred=post_pred)
            print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                  'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time))
            
            if writer is not None:
                writer.add_scalar('val_acc', val_avg_acc, epoch)
            if val_avg_acc > val_acc_max:
                print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                b_new_best = True
                if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                    save_checkpoint(model, epoch, args,
                                    best_acc=val_acc_max,
                                    optimizer=optimizer,
                                    scheduler=scheduler)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_acc_max,
                                filename='model_final.pt')
            if b_new_best:
                print('Copying to model.pt new best model!!!!')
                shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))
            '''

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"当前学习率：{current_lr}")

    print('Training Finished !, Best Accuracy: ', val_acc_max)

    return val_acc_max

