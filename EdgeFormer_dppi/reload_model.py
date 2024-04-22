import time
import argparse
import numpy as np
import torch
import sklearn.metrics as metrics
from torch import nn

from utils import load_test_data
from models import EdgeFormer
from torch.utils import data
from tqdm import tqdm

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', type=str,
                    metavar='DEVICE', help='use CUDA if available')
parser.add_argument('--dataset', type=str, default='test',
                    help='the name of dataset')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--test_dataset_path', default=None, type=str,
                    help='test_dataset_path')
args = parser.parse_args()
args.cuda = True
dataset = args.dataset

def test_one_point_cloud(model, test_dataset_path, dataset):
    path = test_dataset_path + "/" + dataset + "/"
    print(path)
    features1_dppi, features2_dpip, labels = load_test_data(path=path, dataset=dataset)
    if args.cuda:
        model.cuda()
        features1_dppi = features1_dppi.cuda()
        features2_dpip = features2_dpip.cuda()
        labels = labels.cuda()

    torch_dataset = data.TensorDataset(features1_dppi, features2_dpip, labels)
    test_loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,  # 每批提取的数量
        shuffle=False,  # 要不要打乱数据（打乱比较好）
        num_workers=0,  # 多少线程来读取数据
    )

    model.eval()
    test_true = []
    test_pred = []
    test_loss = 0.0
    count = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for (batch_features1_dppi, batch_features2_dpip, batch_labels) in tqdm(test_loader, desc=f'Test'):
            batch_features1_dppi = batch_features1_dppi.to(args.device)
            batch_features2_dpip = batch_features2_dpip.to(args.device)
            batch_labels = batch_labels.to(args.device)

            output = model(batch_features1_dppi, batch_features2_dpip)
            loss = criterion(output, batch_labels)
            preds = output.max(dim=1)[1]

            count += batch_features1_dppi.size(0)
            test_loss += loss.item() * args.batch_size
            test_true.append(batch_labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_recall = metrics.recall_score(test_true, test_pred)
    test_precision = metrics.precision_score(test_true, test_pred)
    test_jaccard = metrics.jaccard_score(test_true, test_pred)
    test_MCC = metrics.matthews_corrcoef(test_true, test_pred)
    test_MSE_100 = metrics.mean_squared_error(test_true, test_pred)
    roc_auc_score = metrics.roc_auc_score(test_true, test_pred)


    with open(path + "EdgeFormer_dppi_outs.txt", "a+") as fo:
        for i in range(len(test_pred)):
            fo.write(str(test_pred[i].item()))
            fo.write("\n")
    print('Test %s: loss: %.4f, accuracy: %.4f, precision: %.4f, recall: %.4f, jaccard: %.4f, MCC: %.4f, MSE_100: %.4f'
          % (dataset, test_loss / count, test_accuracy, test_precision, test_recall, test_jaccard, test_MCC, test_MSE_100))
    print(roc_auc_score)

if __name__ == '__main__':
    start_time = time.time()
    # 载入模型
    PATH = './checkpoints/EdgeFormer_1e-6_best_model.pkl'
    checkpoint = torch.load(PATH)
    model = EdgeFormer()
    model.load_state_dict(checkpoint)

    # test_dataset_path
    test_one_point_cloud(model, args.test_dataset_path, dataset)

    end_time = time.time()  # 结束计时
    total_time = end_time - start_time
    print(f"Total runtime of the program is {total_time} seconds.")
