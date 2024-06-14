import pandas as pd
import matplotlib.pyplot as plt
import os


def parse_specific_metrics(log_path):
    metrics_of_interest = ['acc', 'f1', 'precision', 'recall', 'mcc']
    metrics = {'epoch': []}
    for metric in metrics_of_interest:
        metrics[f'train_{metric}'] = []
        metrics[f'test_{metric}'] = []

    with open(log_path, 'r') as file:
        for line in file:
            parts = line.split('|')
            if len(parts) < 3:
                continue

            epoch = int(parts[1].split(':')[1].strip())
            metrics['epoch'].append(epoch)

            for part in parts[2:]:
                split_part = part.split(',')
                for metric in split_part:
                    if ':' in metric:
                        key, value = metric.split(':')
                        key = key.strip().replace('train_', '').replace('test_', '').replace(' ', '_')
                        if key in metrics_of_interest:
                            full_key = ('train_' if 'train' in metric else 'test_') + key
                            metrics[full_key].append(float(value.strip()))

    df = pd.DataFrame(metrics)
    df.fillna(method='ffill', inplace=True)
    return df


def find_top_values(series):
    sorted_series = series.sort_values(ascending=False)
    top_values = sorted_series.head(2)
    return top_values.index[0], top_values.iloc[0], top_values.index[1], top_values.iloc[1]


def plot_and_annotate_metrics(df, log_path):
    selected_metrics = ['acc', 'f1', 'precision', 'recall', 'mcc']
    plt.figure(figsize=(15, len(selected_metrics) * 5))

    for i, metric in enumerate(selected_metrics, 1):
        plt.subplot(len(selected_metrics), 1, i)
        train_series = df[f'train_{metric}']
        test_series = df[f'test_{metric}']
        plt.plot(df['epoch'], train_series, label='Train', marker='o')
        plt.plot(df['epoch'], test_series, label='Test', marker='x')
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()

        # Finding and annotating top two maximum values
        train_idx1, train_val1, train_idx2, train_val2 = find_top_values(train_series)
        test_idx1, test_val1, test_idx2, test_val2 = find_top_values(test_series)

        plt.annotate(f'Max Train {metric.capitalize()}: {train_val1:.4f} at Epoch {df["epoch"][train_idx1]}',
                     xy=(df['epoch'][train_idx1], train_val1), xycoords='data',
                     textcoords="offset points", xytext=(0, 10), ha='center')
        plt.annotate(f'2nd Max Train {metric.capitalize()}: {train_val2:.4f} at Epoch {df["epoch"][train_idx2]}',
                     xy=(df['epoch'][train_idx2], train_val2), xycoords='data',
                     textcoords="offset points", xytext=(0, -15), ha='center')

        plt.annotate(f'Max Test {metric.capitalize()}: {test_val1:.4f} at Epoch {df["epoch"][test_idx1]}',
                     xy=(df['epoch'][test_idx1], test_val1), xycoords='data',
                     textcoords="offset points", xytext=(0, 10), ha='center')
        plt.annotate(f'2nd Max Test {metric.capitalize()}: {test_val2:.4f} at Epoch {df["epoch"][test_idx2]}',
                     xy=(df['epoch'][test_idx2], test_val2), xycoords='data',
                     textcoords="offset points", xytext=(0, -15), ha='center')

    plt.tight_layout()
    plot_file_path = os.path.join(os.path.dirname(log_path), 'selected_metrics.png')
    plt.savefig(plot_file_path)
    plt.show()
    return plot_file_path



# Usage
#log_path = '/home/z649789/Documents/movement_prediction/log/train/ljmu_seft_v5_train_nmsg_5_nword_40_lr_0.001_heads_2_hidden_dim_20_num_layers_20/log.txt'
log_path = '/home/z649789/Documents/movement_prediction/log/train/ljmu_seft_v4_train_nmsg_5_nword_40_lr_0.001_heads_2_hidden_dim_20_num_layers_20/log.txt'

df_metrics = parse_specific_metrics(log_path)
plot_file_path = plot_and_annotate_metrics(df_metrics, log_path)
