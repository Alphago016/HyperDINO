import numpy as np
import scipy.io as sio


def load_hsi_dataset(dataset_name, image_mat_path, gt_mat_path):
    """
    :param dataset_name: 数据集名称，支持 Houston / PaviaU  / WHU-Hi-LongKou / Salinas /
    :param image_mat_path: 图像mat文件路径
    :param gt_mat_path: 标签mat文件路径
    :return: image (H, W, C), labels (H, W)
    """
    dataset_key_mapping = {
        "Houston": ("data", "groundT"),
        "PaviaU": ("data", "groundT"),
        "WHU-Hi-HanChuan": ("data", "groundT"),
        "WHU-Hi-LongKou": ("data", "groundT"),
        "Salinas": ("data", "groundT"),
        "KSC": ("data", "groundT")
    }

    if dataset_name not in dataset_key_mapping:
        raise ValueError(f"不支持的数据集：{dataset_name}，支持的数据集：{list(dataset_key_mapping.keys())}")

    img_key, gt_key = dataset_key_mapping[dataset_name]

    try:
        img_data = sio.loadmat(image_mat_path)
        image = img_data[img_key]
    except KeyError:
        raise KeyError(
            f"图像mat文件 {image_mat_path} 中未找到key: {img_key}！\n该文件包含的所有key：{list(img_data.keys())}")

    try:
        gt_data = sio.loadmat(gt_mat_path)
        labels = gt_data[gt_key]
    except KeyError:
        raise KeyError(
            f"标签mat文件 {gt_mat_path} 中未找到key: {gt_key}！\n该文件包含的所有key：{list(gt_data.keys())}")

    if len(image.shape) == 3:
        possible_c_indices = []
        for i in range(3):
            dim = image.shape[i]
            if 10 <= dim <= 500:
                possible_c_indices.append(i)

        if len(possible_c_indices) == 1:
            c_idx = possible_c_indices[0]
        else:
            c_idx = np.argmin(image.shape)

        if c_idx == 0:
            image = np.transpose(image, (1, 2, 0))
        elif c_idx == 1:
            image = np.transpose(image, (0, 2, 1))
    elif len(image.shape) == 2:
        H, W = image.shape
        image = image.reshape(H, W, 1)

    labels = labels.squeeze()

    print(f"成功加载{dataset_name}数据集：")
    print(f"  - 图像形状 {image.shape} (H={image.shape[0]}, W={image.shape[1]}, 光谱维度C={image.shape[2]})")
    print(f"  - 标签形状 {labels.shape}")
    return image, labels
