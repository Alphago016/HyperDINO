import numpy as np


def calculate_kappa(confusion_matrix):
    total = np.sum(confusion_matrix)
    if total == 0:
        return 0.0
    po = np.trace(confusion_matrix) / total
    true_sum = np.sum(confusion_matrix, axis=1)
    pred_sum = np.sum(confusion_matrix, axis=0)
    pe = np.sum(true_sum * pred_sum) / (total ** 2)
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 1.0


def write_result_log(log_content, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(log_content)
    print(f"结果日志已保存至：{save_path}")
