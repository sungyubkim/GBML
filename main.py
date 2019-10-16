import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict
import os

from torchmeta.datasets import MiniImagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter

from gbml.maml import MAML
from gbml.reptile import Reptile
from gbml.cavia import CAVIA
from utils import set_seed, set_gpu, check_dir, dict2tsv, BestTracker

def train(args, model, dataloader):

    loss_list = []
    acc_list = []
    grad_list = []
    with tqdm(dataloader, total=args.num_train_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, acc_log, grad_log = model.outer_loop(batch, is_train=True)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            grad_list.append(grad_log)
            pbar.set_description('loss = {:.4f} || acc={:.4f} || grad={:.4f}'.format(np.mean(loss_list), np.mean(acc_list), np.mean(grad_list)))
            if batch_idx >= args.num_train_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)
    grad = np.round(np.mean(grad_list), 4)

    return loss, acc, grad

@torch.no_grad()
def valid(args, model, dataloader):

    loss_list = []
    acc_list = []
    with tqdm(dataloader, total=args.num_valid_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):

            loss_log, acc_log = model.outer_loop(batch, is_train=False)

            loss_list.append(loss_log)
            acc_list.append(acc_log)
            pbar.set_description('loss = {:.4f} || acc={:.4f}'.format(np.mean(loss_list), np.mean(acc_list)))
            if batch_idx >= args.num_valid_batches:
                break

    loss = np.round(np.mean(loss_list), 4)
    acc = np.round(np.mean(acc_list), 4)

    return loss, acc

@BestTracker
def run_epoch(epoch, args, model, train_loader, valid_loader, test_loader):

    res = OrderedDict()
    print('Epoch {}'.format(epoch))
    train_loss, train_acc, train_grad = train(args, model, train_loader)
    valid_loss, valid_acc = valid(args, model, valid_loader)
    test_loss, test_acc = valid(args, model, test_loader)

    res['epoch'] = epoch
    res['train_loss'] = train_loss
    res['train_acc'] = train_acc
    res['train_grad'] = train_grad
    res['valid_loss'] = valid_loss
    res['valid_acc'] = valid_acc
    res['test_loss'] = test_loss
    res['test_acc'] = test_acc
    
    return res

def main(args):

    if args.alg=='MAML':
        model = MAML(args)
    elif args.alg=='Reptile':
        model = Reptile(args)
    elif args.alg=='CAVIA':
        model = CAVIA(args)
    else:
        raise ValueError('Not implemented Meta-Learning Algorithm')

    if args.load:
        model.load()

    train_dataset = MiniImagenet(args.data_path, num_classes_per_task=args.num_way,
                        meta_split='train', 
                        transform=transforms.Compose([
                        transforms.RandomCrop(80, padding=8),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225])),
                        ]),
                        target_transform=Categorical(num_classes=args.num_way)
                        )
    train_dataset = ClassSplitter(train_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    train_loader = BatchMetaDataLoader(train_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    valid_dataset = MiniImagenet(args.data_path, num_classes_per_task=args.num_way,
                        meta_split='val', 
                        transform=transforms.Compose([
                        transforms.CenterCrop(80),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225]))
                        ]),
                        target_transform=Categorical(num_classes=args.num_way)
                        )
    valid_dataset = ClassSplitter(valid_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    valid_loader = BatchMetaDataLoader(valid_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    test_dataset = MiniImagenet(args.data_path, num_classes_per_task=args.num_way,
                        meta_split='test', 
                        transform=transforms.Compose([
                        transforms.CenterCrop(80),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            np.array([0.485, 0.456, 0.406]),
                            np.array([0.229, 0.224, 0.225]))
                        ]),
                        target_transform=Categorical(num_classes=args.num_way)
                        )
    test_dataset = ClassSplitter(test_dataset, shuffle=True, num_train_per_class=args.num_shot, num_test_per_class=args.num_query)
    test_loader = BatchMetaDataLoader(test_dataset, batch_size=args.batch_size,
        shuffle=True, pin_memory=True, num_workers=args.num_workers)

    for epoch in range(args.num_epoch):

        res, is_best = run_epoch(epoch, args, model, train_loader, valid_loader, test_loader)
        dict2tsv(res, os.path.join(args.result_path, args.alg, args.log_path))

        if is_best:
            model.save()
        torch.cuda.empty_cache()

        # model.lr_sched()

    return None

def parse_args():
    import argparse

    parser = argparse.ArgumentParser('Gradient-Based Meta-Learning Algorithms')
    # experimental settings
    parser.add_argument('--seed', type=int, default=2020,
        help='Random seed.')   
    parser.add_argument('--data_path', type=str, default='./',
        help='Path of MiniImagenet.')
    parser.add_argument('--result_path', type=str, default='./res')
    parser.add_argument('--log_path', type=str, default='result.tsv')
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    parser.add_argument('--load', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--load_path', type=str, default='best_model.pth')
    parser.add_argument('--device', type=int, nargs='+', default=[0], help='0 = CPU.')
    parser.add_argument('--num_workers', type=int, default=4,
        help='Number of workers for data loading (default: 4).')
    # training settings
    parser.add_argument('--num_epoch', type=int, default=400,
        help='Number of epochs for meta train.') 
    parser.add_argument('--batch_size', type=int, default=4,
        help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--num_train_batches', type=int, default=250,
        help='Number of batches the model is trained over (default: 250).')
    parser.add_argument('--num_valid_batches', type=int, default=150,
        help='Number of batches the model is trained over (default: 150).')
    # meta-learning settings
    parser.add_argument('--num_shot', type=int, default=1,
        help='Number of support examples per class (k in "k-shot", default: 1).')
    parser.add_argument('--num_query', type=int, default=15,
        help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--num_way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--alg', type=str, default='MAML')
    # algorithm settings
    parser.add_argument('--n_inner', type=int, default=5)
    parser.add_argument('--inner_lr', type=float, default=1e-2)
    parser.add_argument('--outer_lr', type=float, default=1e-3)
    parser.add_argument('--outer_opt', type=str, default='Adam')
    # network settings
    parser.add_argument('--net', type=str, default='ConvNet')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    set_gpu(args.device)
    check_dir(args)
    main(args)