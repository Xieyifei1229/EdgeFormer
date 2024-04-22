import os
import random
import time
import argparse
import numpy as np

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm
import sklearn.metrics as metrics

from utils import back_loader
from models import EdgeFormer

# settings for training
parser = argparse.ArgumentParser(description='EdgeFormer')
parser.add_argument('--device', default='cuda:0', type=str,
                    metavar='DEVICE', help='use CUDA if available')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--best_val_loss', type=float, default=float('inf'),
                    help='best_val_loss.')
# 超参
parser.add_argument('--lr', type=float, default=1e-6,
                     help='Initial learning rate.')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--milestones', type=float, default=[75, 150],
                    help='milestones')
parser.add_argument('--gamma', type=float, default=0.1,
                   help='lr decays to gamma * lr every decay epoch')

args = parser.parse_args()

args.cuda = True


# settings for seed
torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

class Stats:
    def __init__(self):
        self.losses = []
        self.accuracies = []
        self.recalls = []
        self.precisions = []
        self.jaccardes = []
        self.MCCes = []
        self.MSE_100es = []

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

def train_one_epoch(model, optimizer, scheduler, train_loader):
    model.train()
    train_loss = 0.0
    count = 0
    train_epoch_true = []
    train_epoch_pred = []

    criterion = nn.CrossEntropyLoss()

    for (batch_features1_dppi, batch_features2_dpip, batch_labels) in tqdm(train_loader,
                                                                           desc=f'Train Epoch {epoch + 1}/{args.epochs}'):
        batch_features1_dppi = batch_features1_dppi.to(args.device)
        batch_features2_dpip = batch_features2_dpip.to(args.device)
        batch_labels = batch_labels.to(args.device)

        # 将模型转为训练模式，并将优化器梯度置0
        optimizer.zero_grad()
        # 计算输出时，对所有的节点计算输出
        output = model(batch_features1_dppi, batch_features2_dpip)
        # 损失函数
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()

        preds = output.max(dim=1)[1]

        count += args.batch_size
        train_loss += loss.item() * args.batch_size
        train_epoch_true.append(batch_labels.cpu().numpy())
        train_epoch_pred.append(preds.detach().cpu().numpy())

    scheduler.step()
    print("lr: " + str(optimizer.param_groups[0]['lr']))
    train_epoch_true = np.concatenate(train_epoch_true)
    train_epoch_pred = np.concatenate(train_epoch_pred)

    epoch_loss = train_loss * 1.0 / count
    epoch_acc = metrics.accuracy_score(train_epoch_true, train_epoch_pred)
    epoch_recall = metrics.recall_score(train_epoch_true, train_epoch_pred)
    epoch_precision = metrics.precision_score(train_epoch_true, train_epoch_pred)
    epoch_jaccard = metrics.jaccard_score(train_epoch_true, train_epoch_pred)
    epoch_MCC = metrics.matthews_corrcoef(train_epoch_true, train_epoch_pred)
    epoch_MSE_100 = metrics.mean_squared_error(train_epoch_true, train_epoch_pred) * 100

    return epoch_loss, epoch_acc, epoch_recall, epoch_precision, epoch_jaccard, epoch_MCC, epoch_MSE_100

def train(model, epoch, optimizer, scheduler, train_loader, val_loader, train_stats, val_stats, boardio, textio):

    train_one_loss, train_one_acc, train_one_recall, train_one_precision, train_one_jaccard, train_one_MCC, train_one_MSE_100 = train_one_epoch(model, optimizer, scheduler, train_loader)
    # 在验证集上进行性能评估
    val_loss, val_acc, val_recall, val_precision, val_jaccard, val_MCC, val_MSE_100 = validation(model, val_loader)

    # 保存train_stats
    train_stats.losses.append(train_one_loss)
    train_stats.accuracies.append(train_one_acc)
    train_stats.recalls.append(train_one_recall)
    train_stats.precisions.append(train_one_precision)
    train_stats.jaccardes.append(train_one_jaccard)
    train_stats.MCCes.append(train_one_MCC)
    train_stats.MSE_100es.append(train_one_MSE_100)

    # 保存val_stats
    val_stats.losses.append(val_loss)
    val_stats.accuracies.append(val_acc)
    val_stats.recalls.append(val_recall)
    val_stats.precisions.append(val_precision)
    val_stats.jaccardes.append(val_jaccard)
    val_stats.MCCes.append(val_MCC)
    val_stats.MSE_100es.append(val_MSE_100)

    snapshot_path = 'checkpoints/EdgeFormer_1e-6_model.pkl'.format(epoch + 1)
    snap = {'epoch': epoch + 1,
            'model': model.state_dict(),
            'min_loss': val_loss,  # Use validation loss for saving
            'optimizer': optimizer.state_dict()}
    torch.save(snap, snapshot_path)
    print(f'Model saved at {snapshot_path}')

    if val_loss < args.best_val_loss:
        print("update val_loss: " + str(val_loss) + "   " + str(args.best_val_loss))
        args.best_val_loss = val_loss
        # 保存模型
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': args.best_val_loss,
                'optimizer': optimizer.state_dict(), }
        torch.save(snap, 'checkpoints/EdgeFormer_1e-6_best_model.pkl')
        torch.save(model.state_dict(), 'checkpoints/EdgeFormer_1e-6_best_model.pkl')

    boardio.add_scalar('Train_Loss', train_one_loss, epoch + 1)
    boardio.add_scalar('Test_Loss', val_loss, epoch + 1)
    boardio.add_scalar('Best_Test_Loss', args.best_val_loss, epoch + 1)

    textio.cprint('Train: Epoch %d, loss: %.4f, accuracy: %.4f, precision: %.4f, recall: %.4f, jaccard: %.4f, MCC: %.4f, MSE_100: %.4f  Val: loss: %.4f, accuracy: %.4f, precision: %.4f, recall: %.4f, jaccard: %.4f, MCC: %.4f, MSE_100: %.4f'
                  % (epoch + 1, train_one_loss, train_one_acc, train_one_precision, train_one_recall, train_one_jaccard, train_one_MCC, train_one_MSE_100,
                     val_loss, val_acc, val_precision, val_recall, val_jaccard, val_MCC, val_MSE_100))

def validation(model, val_loader):
    model.eval()
    val_true = []
    val_pred = []
    val_loss = 0.0
    count = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for (batch_features1_dppi, batch_features2_dpip, batch_labels) in tqdm(val_loader, desc=f'Validation'):
            batch_features1_dppi = batch_features1_dppi.to(args.device)
            batch_features2_dpip = batch_features2_dpip.to(args.device)
            batch_labels = batch_labels.to(args.device)

            output = model(batch_features1_dppi, batch_features2_dpip)
            loss = criterion(output, batch_labels)
            preds = output.max(dim=1)[1]

            count += batch_features1_dppi.size(0)
            val_loss += loss.item() * args.batch_size
            val_true.append(batch_labels.cpu().numpy())
            val_pred.append(preds.detach().cpu().numpy())

    val_true = np.concatenate(val_true)
    val_pred = np.concatenate(val_pred)

    val_accuracy = metrics.accuracy_score(val_true, val_pred)
    val_recall = metrics.recall_score(val_true, val_pred)
    val_precision = metrics.precision_score(val_true, val_pred)
    val_jaccard = metrics.jaccard_score(val_true, val_pred)
    val_MCC = metrics.matthews_corrcoef(val_true, val_pred)
    val_MSE_100 = metrics.mean_squared_error(val_true, val_pred) * 100

    return val_loss / count, val_accuracy, val_recall, val_precision, val_jaccard, val_MCC, val_MSE_100

if __name__ == '__main__':
    t_total = time.time()

    # 设置logs
    # tensorboard --logdir='./checkpoints/logs'
    boardio = SummaryWriter(log_dir='./checkpoints/logs')
    textio = IOStream('checkpoints/logs/run_EdgeFormer_1e-6.log')
    textio.cprint(str(args))

    # 加载数据集num_workers=4
    train_loader = back_loader('train', args.batch_size)
    val_loader = back_loader('val', args.batch_size)

    # Model and optimizer
    args.device = torch.device(args.device)

    model = EdgeFormer()
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    train_stats = Stats()
    val_stats = Stats()

    # 指定保存检查点的路径
    # checkpoint_path = 'checkpoints/EdgeFormer_1e-6_StepLR_model.pkl'
    #
    # if os.path.isfile(checkpoint_path):
    #     # 加载检查点
    #     checkpoint = torch.load(checkpoint_path)
    #     start_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     args.best_val_loss = checkpoint['min_loss']
    #
    #     print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")

    for epoch in range(args.epochs):
        train(model, epoch, optimizer, scheduler, train_loader, val_loader, train_stats, val_stats, boardio, textio)
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
