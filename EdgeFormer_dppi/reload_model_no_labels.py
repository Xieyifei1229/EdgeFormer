import argparse
import numpy as np
import torch

from utils import load_test_data_no_labels
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
    features1_dppi, features2_dpip = load_test_data_no_labels(path=path, dataset=dataset)
    if args.cuda:
        model.cuda()
        features1_dppi = features1_dppi.cuda()
        features2_dpip = features2_dpip.cuda()

    torch_dataset = data.TensorDataset(features1_dppi, features2_dpip)
    test_loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=args.batch_size,  # 每批提取的数量
        shuffle=False,  # 要不要打乱数据（打乱比较好）
        num_workers=0,  # 多少线程来读取数据
    )

    model.eval()
    test_pred = []
    count = 0.0

    with torch.no_grad():
        for (batch_features1_dppi, batch_features2_dpip) in tqdm(test_loader, desc=f'Test'):
            batch_features1_dppi = batch_features1_dppi.to(args.device)
            batch_features2_dpip = batch_features2_dpip.to(args.device)

            output = model(batch_features1_dppi, batch_features2_dpip)
            preds = output.max(dim=1)[1]

            count += batch_features1_dppi.size(0)
            test_pred.append(preds.detach().cpu().numpy())

    test_pred = np.concatenate(test_pred)

    with open(path + "EdgeFormer_dppi_PartNet_outs.txt", "a+") as fo:
        for i in range(len(test_pred)):
            fo.write(str(test_pred[i].item()))
            fo.write("\n")

if __name__ == '__main__':
    # 载入模型
    PATH = './checkpoints/EdgeFormer_1e-6_best_model.pkl'
    checkpoint = torch.load(PATH)
    model = EdgeFormer()
    model.load_state_dict(checkpoint)

    test_one_point_cloud(model, args.test_dataset_path, dataset)
