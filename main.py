from __future__ import division
from __future__ import print_function
import os, sys, shutil, time, random
import argparse
import torch
from torch.nn import Parameter
from torch.nn.parallel import data_parallel
from torch.nn.parallel.scatter_gather import scatter_kwargs
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, Cutout, DataIter
from torch.utils.data.sampler import SubsetRandomSampler
import models
import numpy as np
import random
from topology import Adjacency, RNNAdjacency

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Training script for Networks with Soft Sharing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Data / Model
parser.add_argument('data_path', metavar='DPATH', type=str, help='Path to dataset')
parser.add_argument('--dataset', metavar='DSET', type=str, choices=['cifar10', 'cifar100', 'imagenet'], help='Choose between CIFAR/ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='swrn', help='model architecture: ' + ' | '.join(model_names) + ' (default: shared wide resnet)')
parser.add_argument('--depth', type=int, metavar='N', default=28)
parser.add_argument('--wide', type=int, metavar='N', default=2)
parser.add_argument('--bank_size', type=int, default=2, help='Size of filter bank for soft shared network')

# Optimization
parser.add_argument('--epochs', metavar='N', type=int, default=250, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120, 160], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2], help='LR is multiplied by gamma on schedule')

#Regularization
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--cutout', dest='cutout', action='store_true', help='Enable cutout augmentation')

# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='Print frequency, minibatch-wise (default: 200)')
parser.add_argument('--save_path', type=str, default='./snapshots/', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate model on test set')

# Acceleration
parser.add_argument('--ngpu', type=int, default=8, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers (default: 16)')

# Random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--job-id', type=str, default='')

# Adj setting
# parser.add_argument('--adj_n_components', type=int, default=1, help='# of components in the mixture distribution of adj')
# parser.add_argument('--adj_feature_dim', type=int, default=10, help='dimension of node features to build adj')
# parser.add_argument('--adj_hard', dest='adj_hard', action='store_true', help='whether using hard adj')
# parser.add_argument('--adj_gradient_estimator', type=str, default='gsm', help='gradient estimator to optimize adj')
# parser.add_argument('--tau0', type=float, default=1., help='tau0')
# parser.add_argument('--tau_min', type=float, default=1., help='tau_min')
# parser.add_argument('--tau_anneal_rate', type=float, default=0.00003, help='tau_anneal_rate')
parser.add_argument('--adj_lr', type=float, default=3e-3, help='learning rate for adj')
parser.add_argument('--entropy_weight', type=float, default=1., help='entropy_weight for adj')
parser.add_argument('--adj_update_steps_per_epoch', type=int, default=100, help='interval for updating adj rnn')
parser.add_argument('--adj_batch_size', type=int, default=16, help='batch size for updating adj rnn')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
assert args.adj_batch_size % args.ngpu == 0
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(args.ngpu)])
print(torch.cuda.device_count())
job_id = args.job_id
args.save_path = args.save_path + job_id
result_png_path = './results/' + job_id + '.png'
if not os.path.isdir('results'): os.mkdir('results')

out_str = str(args)
print(out_str)

if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda: torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

best_acc = 0

def load_dataset():
    if args.dataset == 'cifar10':
        mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = dset.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        mean, std = [x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]]
        dataset = dset.CIFAR100
        num_classes = 100
    elif args.dataset != 'imagenet': assert False, "Unknown dataset : {}".format(args.dataset)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
        if args.cutout: train_transform.transforms.append(Cutout(n_holes=1, length=16))
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        valid_iter = None

        if args.evaluate:
            train_data = dataset(args.data_path, train=True, transform=train_transform, download=True)
            test_data = dataset(args.data_path, train=False, transform=test_transform, download=True)

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        else:
            # partition training set into two instead. note that test_data is defined using train=True
            train_data = dataset(args.data_path, train=True, transform=train_transform, download=True)
            valid_data = dataset(args.data_path, train=True, transform=test_transform, download=True)

            indices = list(range(len(train_data)))
            np.random.shuffle(indices)
            split = int(0.9 * len(train_data))
            train_indices, valid_indices = indices[:split], indices[split:]
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, sampler=SubsetRandomSampler(train_indices))
            valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, sampler=SubsetRandomSampler(valid_indices), drop_last=True)
            valid_iter = DataIter(valid_loader)
            # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)

            test_data = dataset(args.data_path, train=False, transform=test_transform, download=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


    elif args.dataset == 'imagenet':
        import imagenet_seq
        train_loader = imagenet_seq.data.Loader('train', batch_size=args.batch_size, num_workers=args.workers)
        test_loader = imagenet_seq.data.Loader('val', batch_size=args.batch_size, num_workers=args.workers)
        num_classes = 1000
    else: assert False, 'Do not support dataset : {}'.format(args.dataset)

    return num_classes, train_loader, valid_iter, test_loader


def load_model(num_classes, log):
    print_log("=> creating model '{}'".format(args.arch), log)

    net = models.__dict__[args.arch](args.depth, args.wide, args.bank_size, num_classes)
    print_log("=> network :\n {}".format(net), log)
    net = net.cuda()
    trainable_params = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([p.numel() for p in trainable_params])
    print_log("Number of parameters: {}".format(params), log)
    return net


def main():
    global best_acc

    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("PyTorch  version : {}".format(torch.__version__), log)
    print_log("CuDNN  version : {}".format(torch.backends.cudnn.version()), log)

    if not os.path.isdir(args.data_path): os.makedirs(args.data_path)

    num_classes, train_loader, valid_iter, test_loader = load_dataset()
    net = load_model(num_classes, log)
    net_p = torch.nn.DataParallel(net, device_ids=range(args.ngpu))

    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.bank_size > 0:
        coefficients =[
            Parameter(torch.zeros((net.stage_1.nlayers,)+net.stage_1.bank.coefficient_shape).cuda()),
            Parameter(torch.zeros((net.stage_2.nlayers,)+net.stage_2.bank.coefficient_shape).cuda()),
            Parameter(torch.zeros((net.stage_3.nlayers,)+net.stage_3.bank.coefficient_shape).cuda())]
        for item in coefficients:
            coefficient_inits = torch.zeros_like(item.data)
            torch.nn.init.orthogonal_(coefficient_inits)
            item.data = coefficient_inits
            # item.data = torch.eye(item.shape[0]).view_as(item).cuda()
    else:
        coefficients = None
    params = group_weight_decay(net, state['decay'], coefficients)
    optimizer = torch.optim.SGD(params, state['learning_rate'], momentum=state['momentum'], nesterov=(state['momentum'] > 0.0))

    adjacency = RNNAdjacency(3, net.stage_1.nlayers, lr=args.adj_lr, entropy_weight=args.entropy_weight)
    # adjacency.load()
    # adjacency = Adjacency([net.module.stage_1.nlayers+1, net.module.stage_2.nlayers+1, net.module.stage_3.nlayers+1],
    #                     args.adj_n_components, args.adj_feature_dim, args.tau0, args.tau_min, args.tau_anneal_rate,
    #                     args.adj_hard, args.adj_gradient_estimator, args.adj_lr)
    rl_schedule = {}
    rl_seg_list = [len(train_loader) // args.adj_update_steps_per_epoch] * args.adj_update_steps_per_epoch
    for _i in range(len(train_loader) % args.adj_update_steps_per_epoch):
        rl_seg_list[_i] += 1
    for j in range(1, len(rl_seg_list)):
        rl_seg_list[j] += rl_seg_list[j - 1]
    for j in rl_seg_list:
        rl_schedule[j - 1] = 1

    recorder = RecorderMeter(args.epochs)
    if args.resume:
        if args.resume == 'auto': args.resume = os.path.join(args.save_path, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            recorder.refresh(args.epochs)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            coefficients = checkpoint['coefficients']
            adjacency = checkpoint['adjacency']
            best_acc = recorder.max_accuracy(False)
            print_log("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.evaluate:
        validate(test_loader, net, criterion, log, adjacency, coefficients)
        return

    start_time = time.time()
    epoch_time = AverageMeter()
    train_los = -1

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, train_los)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                    + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        train_acc, train_los = train(train_loader, valid_iter, net, criterion, optimizer, epoch, log, adjacency, coefficients, args.adj_batch_size, args.ngpu, rl_schedule, net_p)

        val_acc, val_los   = validate(test_loader, net, criterion, log, adjacency, coefficients)
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        if epoch % 5 == 0:
            adjacency.eval()
            with torch.no_grad():
                samp = adjacency.sample(deterministic=True)
            for s in samp:
                print(s.data.cpu().numpy())

        is_best = False
        if val_acc > best_acc:
            is_best = True
            best_acc = val_acc

        save_checkpoint({
          'epoch': epoch + 1,
          'arch': args.arch,
          'state_dict': net.state_dict(),
          'recorder': recorder,
          'optimizer' : optimizer.state_dict(),
          'coefficients': coefficients,
          'adjacency': adjacency,
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(result_png_path)
    log.close()


def train(train_loader, valid_iter, model, criterion, optimizer, epoch, log, adjacency, coefficients, adj_batch_size, ngpu, rl_schedule, net_p):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    rl_time = AverageMeter()
    losses = AverageMeter()
    ent_losses = AverageMeter()
    rl_losses = AverageMeter()
    rewards = AverageMeter()
    sample_vars = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    adjacency.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            adjacencies = adjacency.sample(deterministic=False if epoch < 200 else True)
        output = model(input, adjacencies=adjacencies, coefficients=coefficients)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch <= 2 or (epoch >= 50 and epoch < 200):
            for j in range(rl_schedule.get(i, 0)):
                rl_start_time = time.time()
                adjacency.zero_grad()
                samp = adjacency.sample(adj_batch_size)
                sample_vars.update(sum([item[1:].tril().sum().item() for item in samp.var(0).chunk(3)])/63.)
                with torch.no_grad():
                    if epoch < 50:
                        rs = torch.zeros(adj_batch_size).cuda()
                    else:
                        input_search, target_search = valid_iter.next_batch
                        input_search = input_search.repeat(ngpu,1,1,1)
                        target_search = target_search.cuda(non_blocking=True)
                        output_search = []
                        for k in range(adj_batch_size//ngpu):
                            output_search.append(net_p(input_search, samp[k*ngpu:(k+1)*ngpu]))
                        rs = torch.cat(output_search, 0).max(1)[1].eq(target_search.repeat(adj_batch_size)).view(
                             adj_batch_size, -1).float().mean(1)
                adjacency.step(rs, epoch)
                rl_time.update(time.time() - rl_start_time)
                ent_losses.update(adjacency.neg_entropy.item(), adj_batch_size)
                rl_losses.update(adjacency.rl_loss.item(), adj_batch_size)
                rewards.update(adjacency.avg_reward.item(), adj_batch_size)

        batch_time.update(time.time() - end)
        end = time.time()
        if i == len(train_loader) - 1: #i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'RL time {rl_time.val:.3f} ({rl_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Reward {reward.val:.4f} ({reward.avg:.4f})   '
                        'Sample variance {sample_var.val:.4f} ({sample_var.avg:.4f})   '
                        'RL loss {rl_loss.val:.4f} ({rl_loss.avg:.4f})   '
                        'Ent loss {ent_loss.val:.4f} ({ent_loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        rl_time=rl_time, loss=losses, reward=rewards, sample_var=sample_vars,
                        rl_loss=rl_losses, ent_loss=ent_losses, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log, adjacency, coefficients):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    adjacency.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input, adjacencies=adjacency.sample(deterministic=True), coefficients=coefficients)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    print_log('  **Test**  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss {losses.avg:.5f} '.format(top1=top1, top5=top5, error1=100-top1.avg, losses=losses), log)
    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule, loss):
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step): lr = lr * gamma
        else: break
    for param_group in optimizer.param_groups: param_group['lr'] = lr
    return lr


def group_weight_decay(net, weight_decay, coefficients):
    decay = []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue
        decay.append(param)
    return [{'params': decay, 'weight_decay': weight_decay}] + ([{'params': coefficients, 'weight_decay': 0.}] if coefficients else [])


def accuracy(output, target, topk=(1,)):
    if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)

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

if __name__ == '__main__': main()
