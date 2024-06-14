import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_specific_metrics_updated(log_path):
    metrics_of_interest = ['train_mle', 'test_mles']
    metrics = {'epoch': []}
    for metric in metrics_of_interest:
        metrics[metric] = []

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
                        key = key.strip().replace(' ', '_')
                        if key in metrics_of_interest:
                            metrics[key].append(float(value.strip()))

    df = pd.DataFrame(metrics)
    df.fillna(method='ffill', inplace=True)
    return df

def plot_and_annotate_metrics_updated_labels(df, log_path):
    selected_metrics = ['train_mle', 'test_mles']
    plt.figure(figsize=(15, len(selected_metrics) * 5))

    for i, metric in enumerate(selected_metrics, 1):
        plt.subplot(len(selected_metrics), 1, i)
        metric_series = df[metric]
        label = 'Train Mean Loss' if 'train' in metric else 'Test Mean Loss'
        plt.plot(df['epoch'], metric_series, label=label, marker='o')
        plt.title(f'{label} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(label)
        plt.legend()

        # Finding and annotating minimum values
        min_val = metric_series.min()
        min_epoch = df['epoch'][metric_series.idxmin()]

        plt.annotate(f'Min {label}: {min_val:.4f} at Epoch {min_epoch}',
                     xy=(min_epoch, min_val), xytext=(min_epoch, min_val),
                     textcoords="offset points", xycoords='data', ha='right', va='bottom', fontsize=9,
                     arrowprops=dict(arrowstyle="->"))

    plt.tight_layout()
    plot_file_path = os.path.join(os.path.dirname(log_path), 'train_test_loss_visu.png')
    plt.savefig(plot_file_path)
    plt.show()
    return plot_file_path

# Usage with the uploaded file
# Usage
log_path='/home/z649789/Documents/movement_prediction/log/train/ljmu_seft_v5_train_nmsg_5_nword_40_lr_0.001_heads_2_hidden_dim_20_num_layers_20/log.txt'
#log_path='/home/z649789/Documents/movement_prediction/log/train/ljmu_seft_v4_train_nmsg_5_nword_40_lr_0.001_heads_2_hidden_dim_20_num_layers_20/log.txt'
#log_path='/home/z649789/Documents/movement_prediction/log/train/ljmu_seft_v3_train_nmsg_5_nword_40_lr_0.001_heads_2_hidden_dim_20_num_layers_20/log.txt'
#log_path='/home/z649789/Documents/movement_prediction/log/train/ljmu_seft_v2_train_nmsg_5_nword_40_lr_0.001_heads_2_hidden_dim_14_num_layers_14/log.txt'
#log_path = '/home/z649789/Documents/movement_prediction/log/train/ljmu_seft_v1_train_nmsg_5_nword_40_lr_0.001_heads_2_hidden_dim_20_num_layers_20/log.txt'

df_metrics = parse_specific_metrics_updated(log_path)
plot_file_path = plot_and_annotate_metrics_updated_labels(df_metrics, log_path)
plot_file_path
