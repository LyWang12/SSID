import random
import time
import argparse
import copy
import tqdm
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os.path as osp
from transformer import swin_tiny_patch4_window7_224
from network import ImageClassifier
from utils import ContinuousDataloader
from transforms import ResizeImage
from lr_scheduler import LrScheduler
from data_list_index import ImageList
from Loss import *

import torch

# 1. 强制让 is_available 返回 False，这样代码内部的逻辑会倾向于走 CPU 分支
torch.cuda.is_available = lambda: False

# 2. 核心黑魔法：拦截所有的 .cuda() 调用，让它直接返回自身（即留在 CPU 上）
# 这样即使代码里写了 tensor.cuda() 或 model.cuda()，也不会报错
torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.nn.Module.cuda = lambda self, *args, **kwargs: self

# 3. 拦截 torch.load，强制加上 map_location='cpu'
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return original_load(*args, **kwargs)
torch.load = patched_load

print("⚠️ 已启动 CPU 兼容模式：所有的 .cuda() 调用将被忽略，模型将强制在 CPU 运行。")


def get_current_time():
    time_stamp = time.time()
    local_time = time.localtime(time_stamp)
    str_time = time.strftime('%Y-%m-%d_%H-%M-%S', local_time)
    return str_time

def main(args: argparse.Namespace, config):
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.dset == "visda":
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        train_transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
            
    val_tranform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    train_source_dataset = ImageList(open(args.s_dset_path).readlines(), transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=2*args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    memory_source_loader = DataLoader(train_source_dataset, batch_size=1,
                                     shuffle=False, num_workers=args.workers, drop_last=False)

    train_target_dataset = ImageList(open(args.t_dset_path).readlines(), transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)

    val_dataset = ImageList(open(args.t_dset_path).readlines(), transform=val_tranform)
    if args.dset == "visda":
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64)
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = val_loader

    train_source_iter = ContinuousDataloader(train_source_loader)
    train_target_iter = ContinuousDataloader(train_target_loader)

    s_len = train_source_dataset.__len__()
    t_len = val_dataset.__len__()

    # load model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = swin_tiny_patch4_window7_224()
    weights_dict = torch.load('swin_tiny_patch4_window7_224.pth')["model"]
    # 删除有关分类类别的权重
    for k in list(weights_dict.keys()):
        if "head" in k:
            del weights_dict[k]
    print(backbone.load_state_dict(weights_dict, strict=False))

    if args.dset == "office":
        num_classes = 31
    elif args.dset == "office-home":
        num_classes = 65
    elif args.dset == "visda":
        num_classes = 12
    classifier = ImageClassifier(backbone, num_classes).cuda()
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))

    # define optimizer and lr scheduler
    all_parameters = classifier.get_parameters()
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_sheduler = LrScheduler(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)


    # start training
    best_acc1 = 0.
    cls_criterion = Cls_Loss(num_classes).cuda()
    
    memory_source_features = torch.zeros(s_len, 256).cuda()
    memory_source_labels = torch.zeros(s_len).long().cuda()
    memory_inter_mean = torch.zeros(s_len, 768).cuda()
    memory_inter_std = torch.zeros(s_len, 768).cuda()
    epoch_iterator = tqdm.tqdm(memory_source_loader)
    classifier.eval()
    current_time = get_current_time()
    print('Begin memory', 'time=', current_time)
    with torch.no_grad():
        for i, (images, labels, idx) in enumerate(epoch_iterator):
            images = images.cuda()
            labels = labels.cuda()
            # get logit outputs
            output, f, ft = classifier(images)
            memory_source_features[idx] = f
            memory_source_labels[idx] = labels
            mean, std = calc_ins_mean_std(ft)
            memory_inter_mean[idx] = mean.squeeze()
            memory_inter_std[idx] = std.squeeze()
        gc.collect()
    current_time = get_current_time()
    print('End memory', 'time=', current_time)
    print('Memory initial!')
    
    
    
    
    for epoch in range(args.epochs):
        # train for one epoch
        current_time = get_current_time()
        print('Begin training', 'epoch=', epoch, 'time=', current_time)
        train(train_source_iter, train_target_iter, classifier, optimizer,
              lr_sheduler, epoch, args, cls_criterion, num_classes, memory_source_features, memory_source_labels, memory_inter_mean, memory_inter_std)
        current_time = get_current_time()
        print('End training', 'epoch=', epoch, 'time=', current_time)
        # evaluate on validation set
        if args.dset == "visda":
            acc1 = validate_visda(val_loader, classifier, epoch, config)
        else:
            current_time = get_current_time()
            print('Begin testing', 'epoch=', epoch, 'time=', current_time)
            acc1 = validate(val_loader, classifier, args)
            current_time = get_current_time()
            print('End testing', 'epoch=', epoch, 'time=', current_time)
        # remember the best top1 accuracy and checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())
            model = classifier
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_dice': best_acc1
            }
            torch.save(state, args.output)
        best_acc1 = max(acc1, best_acc1)
        print("epoch = {:02d},  acc1={:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1))
        config["out_file"].write("epoch = {:02d},  best_acc1 = {:.3f}, best_acc1 = {:.3f}".format(epoch, acc1, best_acc1) + '\n')
        config["out_file"].flush()

    print("best_acc1 = {:.3f}".format(best_acc1))
    config["out_file"].write("best_acc1 = {:.3f}".format(best_acc1) + '\n')
    config["out_file"].flush()

    # evaluate on test set
    classifier.load_state_dict(best_model)
    if args.dset == "visda":
        acc1 = validate_visda(test_loader, classifier, epoch, config)
    else:
        acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:.3f}".format(acc1))
    config["out_file"].write("test_acc1 = {:.3f}".format(acc1) + '\n')
    config["out_file"].flush()


class KL_Loss(torch.nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, f_s, f_t, pre_t):
        l_t = torch.argmax(pre_t, dim=1)
        kl_div = torch.nn.functional.kl_div(f_t.softmax(dim=-1).log(), f_s[l_t].softmax(dim=-1), reduction='sum')
        b = pre_t.shape[0]
        del l_t
        gc.collect()

        return kl_div / b


def calc_ins_mean_std(x, eps=1e-5):
    size = x.size()
    assert (len(size) == 3)
    N, C = size[0], size[2]
    var = x.contiguous().view(N, -1, C).var(dim=1)  + eps
    std = var.sqrt().view(N, 1, C)
    mean = x.contiguous().view(N, -1, C).mean(dim=1).view(N, 1, C)

    return mean.detach(), std.detach()

def CalculateMean(features, labels, class_num, args):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    avg_CxA = torch.zeros(C, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1.0

    del onehot
    gc.collect()
    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()

def train(train_source_iter: ContinuousDataloader, train_target_iter: ContinuousDataloader, model: ImageClassifier,
       optimizer: SGD, lr_sheduler: LrScheduler, epoch: int, args: argparse.Namespace, cls_criterion, num_classes,
       memory_source_features, memory_source_labels, memory_inter_mean, memory_inter_std):
    # switch to train mode

    model.train()
    tq = tqdm.tqdm(total=args.iters_per_epoch * args.batch_size)
    tq.set_description('train %d' % epoch)

    kl_si = KL_Loss().cuda()
    kl_st = KL_Loss().cuda()

    kl_mean_t = KL_Loss().cuda()
    kl_std_t = KL_Loss().cuda()


    for i in range(args.iters_per_epoch):
        lr_sheduler.step()

        x_s, labels_s, idx_source = next(train_source_iter)
        x_t, _ , idx_target = next(train_target_iter)

        x_s = x_s.cuda()
        x_t = x_t.cuda()
        labels_s = labels_s.cuda()

        # get features and logit outputs
        x = torch.cat((x_s, x_t), dim=0)
        y, f, ft = model(x)

        b = args.batch_size
        y_s, y_i, y_t = y[:b], y[b:2*b], y[2*b:]
        f_s, f_i, f_t = f[:b], f[b:2*b], f[2*b:]
        ft_s, ft_i, ft_t = ft[:b], ft[b:2*b], ft[2*b:]
        memory_source_features[idx_source[:b]] = f_s
        mean_source = CalculateMean(memory_source_features, memory_source_labels, num_classes, args)
        loss_kl_si = kl_si(mean_source, f_i, y_i)
        loss_kl_st = kl_st(mean_source, f_t, y_t)
        print(mean_source.shape, f_t.shape, y_t.shape)

        ft_i_mean, ft_i_std = calc_ins_mean_std(ft_i)

        memory_inter_mean[idx_source[b:2*b]] = ft_i_mean.squeeze()
        memory_inter_std[idx_source[b:2*b]] = ft_i_std.squeeze()

        mean_inter = CalculateMean(memory_inter_mean, memory_source_labels, num_classes, args)
        std_inter = CalculateMean(memory_inter_std, memory_source_labels, num_classes, args)

        x_t_mean, x_t_std = calc_ins_mean_std(ft_t)
        loss_kl_mean_t = kl_mean_t(mean_inter, x_t_mean.squeeze(), y_t)
        loss_kl_std_t = kl_std_t(std_inter, x_t_std.squeeze(), y_t)


        # compute loss
        classifier_loss = nn.CrossEntropyLoss()(y[:2*b], labels_s[:2*b])
        MI_loss = MI(y_t)
        kl = 1

        total_loss = classifier_loss - args.MI * MI_loss + loss_kl_si + kl*loss_kl_st + loss_kl_mean_t + loss_kl_std_t

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        tq.update(args.batch_size)
        tq.set_postfix(loss='%.6f' % (total_loss))

        # print training log
        if i % args.print_freq == 0:
            print("Epoch: [{:02d}][{}/{}]	total_loss:{:.3f}	classifier_loss:{:.3f}	 MI_loss:{:.3f}	 si_loss:{:.3f}	 st_loss:{:.3f}	 mean_t_loss:{:.3f}	 std_t_loss:{:.3f}	 ".format(\
                epoch, i, args.iters_per_epoch, total_loss, classifier_loss, MI_loss, loss_kl_si, loss_kl_st, loss_kl_mean_t, loss_kl_std_t))
    tq.close()

def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    # switch to evaluate mode
    model.eval()
    tbar = tqdm.tqdm(val_loader, desc='\r')
    start_test = True
    with torch.no_grad():
        for i, (images, target, _) in enumerate(tbar):
            images = images.cuda()
            target = target.cuda()

            # get logit outputs
            output, _, _ = model(images)  # 32,31
            if start_test:
                all_output = output.float()
                all_label = target.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, output.float()), 0)
                all_label = torch.cat((all_label, target.float()), 0)
            tbar.set_description()
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        accuracy = accuracy * 100.0
        print(' accuracy:{:.3f}'.format(accuracy))
    return accuracy

def validate_visda(val_loader, model, epoch, config):
    dict = {0: "plane", 1: "bcybl", 2: "bus", 3: "car", 4: "horse", 5: "knife", 6: "mcyle", 7: "person", 8: "plant", \
            9: "sktb", 10: "train", 11: "truck"}
    model.eval()
    with torch.no_grad():
        tick = 0
        subclasses_correct = np.zeros(12)
        subclasses_tick = np.zeros(12)
        for i, (imgs, labels, _) in enumerate(val_loader):
            tick += 1
            imgs = imgs.cuda()
            pred, _ = model(imgs)
            pred = nn.Softmax(dim=1)(pred)
            pred = pred.data.cpu().numpy()
            pred = pred.argmax(axis=1)
            labels = labels.numpy()
            for i in range(pred.size):
                subclasses_tick[labels[i]] += 1
                if pred[i] == labels[i]:
                    subclasses_correct[pred[i]] += 1
        subclasses_result = np.divide(subclasses_correct, subclasses_tick)
        print("Epoch [:02d]:".format(epoch))
        for i in range(12):
            log_str1 = '\t{}----------({:.3f})'.format(dict[i], subclasses_result[i] * 100.0)
            print(log_str1)
            config["out_file"].write(log_str1 + "\n")
        avg = subclasses_result.mean()
        avg = avg * 100.0
        log_avg = '\taverage:{:.3f}'.format(avg)
        print(log_avg)
        config["out_file"].write(log_avg + "\n")
        config["out_file"].flush()
    return avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transferable Semantic Augmentation for Domain Adaptation')
    parser.add_argument('--arch', type=str, default='SWIN_T')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='6', help="device id to run")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'visda', 'office-home'], help="The dataset used")
    parser.add_argument('--s_dset_path', type=str, default='data/list/home/Clipart_65.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='data/list/home/RealWorld_65.txt', help="The target dataset path list")
    parser.add_argument('--output_dir', type=str, default='log/office-home-65/SWIN_T+I+loss', help="output directory of logs")
    parser.add_argument('--seed', type=int, default=2, help="output directory of logs")
    parser.add_argument('--output', type=str, default='log/office-home-65/SWIN_T+I+loss/C-R.pth', help="output directory of model")

    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--batch-size', default=18, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('--MI', type=float, default=0.1, help="MI_loss_tradeoff")
    args = parser.parse_args(args=[])

    config = {}
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    task = args.s_dset_path.split('/')[-1].split('.')[0].split('_')[0] + "-" + \
           args.t_dset_path.split('/')[-1].split('.')[0].split('_')[0]
    config["out_file"] = open(osp.join(args.output_dir, get_current_time() + "_" + task + "_log.txt"), "w")

    # 配置检查
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
        config["out_file"].write(str("{} = {}".format(arg, getattr(args, arg))) + "\n")
    config["out_file"].flush()
    main(args, config)
