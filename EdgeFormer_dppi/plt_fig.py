import re
import matplotlib.pyplot as plt

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_data(lines):
    train_epochs, train_loss, train_accuracy, train_precision, train_recall, train_jaccard, train_MCC, train_MSE_100 = [], [], [], [], [], [], [], []
    val_epochs, val_loss, val_accuracy, val_precision, val_recall, val_jaccard, val_MCC, val_MSE_100 = [], [], [], [], [], [], [], []

    for line in lines:
        if 'Train:' in line:
            parts = re.findall(r'\d+\.\d+|\d+', line)

            train_epochs.append(int(parts[0]))
            train_loss.append(float(parts[1]))
            train_accuracy.append(float(parts[2]))
            train_precision.append(float(parts[3]))
            train_recall.append(float(parts[4]))
            train_jaccard.append(float(parts[5]))
            train_MCC.append(float(parts[6]))
            train_MSE_100.append(float(parts[8]))

            val_epochs.append(int(parts[0]))
            val_loss.append(float(parts[9]))
            val_accuracy.append(float(parts[10]))
            val_precision.append(float(parts[11]))
            val_recall.append(float(parts[12]))
            val_jaccard.append(float(parts[13]))
            val_MCC.append(float(parts[14]))
            val_MSE_100.append(float(parts[16]))
        else:
            print(f"Warning: Could not parse line - {line}")

    return (
        train_epochs, train_loss, train_accuracy, train_precision, train_recall, train_jaccard, train_MCC, train_MSE_100,
        val_epochs, val_loss, val_accuracy, val_precision, val_recall, val_jaccard, val_MCC, val_MSE_100
    )

def plot_curves(train_epochs, train_metric, val_epochs, val_metric, metric_name, ylabel, name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_epochs, train_metric, label=f'Train {metric_name}', marker='o', markersize=0)
    plt.plot(val_epochs, val_metric, label=f'Validation {metric_name}', marker='o', markersize=0)
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig('./figures/' + name + '.jpg', dpi=500)  # 调整这里的dpi值，300是一个常用的高分辨率值
    plt.show()

file_path = './checkpoints/logs/run_EdgeFormer_1e-6.log'
lines = read_file(file_path)
(
    train_epochs, train_loss, train_accuracy, train_precision, train_recall, train_jaccard, train_MCC, train_MSE_100,
    val_epochs, val_loss, val_accuracy, val_precision, val_recall, val_jaccard, val_MCC, val_MSE_100
) = parse_data(lines)

plot_curves(train_epochs, train_loss, val_epochs, val_loss, 'Loss', 'Loss Value', 'Loss')
plot_curves(train_epochs, train_accuracy, val_epochs, val_accuracy, 'Accuracy', 'Accuracy Value', 'Accuracy')
plot_curves(train_epochs, train_precision, val_epochs, val_precision, 'Precision', 'Precision Value', 'Precision')
plot_curves(train_epochs, train_recall, val_epochs, val_recall, 'Recall', 'Recall Value', 'Recall')
plot_curves(train_epochs, train_jaccard, val_epochs, val_jaccard, 'Jaccard', 'Jaccard Value', 'Jaccard')
plot_curves(train_epochs, train_MCC, val_epochs, val_MCC, 'MCC', 'MCC Value', 'MCC')
plot_curves(train_epochs, train_MSE_100, val_epochs, val_MSE_100, 'MSE_100', 'MSE_100 Value', 'MSE_100')
exit(0)
