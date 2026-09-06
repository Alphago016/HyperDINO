import os
import numpy as np
import matplotlib.pyplot as plt

HOUSTON_PALETTE = np.array([
    [0, 0, 0], [0, 255, 255], [0, 0, 128], [224, 255, 224], [154, 205, 50],
    [144, 238, 144], [202, 255, 112], [218, 112, 214], [204, 50, 153],
    [179, 136, 255], [255, 255, 0], [255, 0, 255], [255, 69, 0],
    [0, 255, 255], [0, 255, 0], [0, 0, 255]
], dtype=np.uint8)

PAVIAU_PALETTE = np.array([
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [0, 255, 255], [255, 165, 0], [128, 0, 128], [128, 128, 0]
], dtype=np.uint8)

LONGKOU_PALETTE = np.array([
    [0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255], [255, 0, 0],
    [255, 0, 255], [255, 255, 0], [173, 216, 230], [204, 153, 255], [219, 219, 219]
], dtype=np.uint8)

HANCHUAN_PALETTE = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [192, 192, 192],
    [128, 128, 128],
    [255, 0, 0],
    [0, 255, 0],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 165, 0],
    [75, 0, 130],
    [255, 215, 0]
], dtype=np.uint8)

SALINAS_PALETTE = np.array([
    [0, 0, 0],
    [0, 0, 139],
    [0, 0, 205],
    [0, 0, 255],
    [0, 128, 255],
    [0, 255, 255],
    [128, 224, 255],
    [102, 205, 170],
    [144, 238, 144],
    [154, 205, 50],
    [255, 255, 0],
    [255, 200, 0],
    [255, 140, 0],
    [255, 69, 0],
    [255, 0, 0],
    [178, 34, 34],
    [101, 67, 33]
], dtype=np.uint8)

KSC_PALETTE = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [255, 128, 0],
    [0, 255, 255],
    [0, 128, 0],
    [128, 128, 0],
    [128, 0, 128],
    [0, 0, 255],
    [255, 0, 255],
    [255, 255, 0],
    [128, 0, 0],
    [255, 0, 0],
    [173, 216, 230],
    [211, 211, 211]
], dtype=np.uint8)

PALETTE_MAP = {
    "Houston": HOUSTON_PALETTE,
    "PaviaU": PAVIAU_PALETTE,
    "WHU-Hi-LongKou": LONGKOU_PALETTE,
    "Salinas": SALINAS_PALETTE,
    "WHU-Hi-HanChuan": HANCHUAN_PALETTE,
    "KSC": KSC_PALETTE
}


def label_to_color(pred_map, dataset_name):
    palette = PALETTE_MAP[dataset_name]
    H, W = pred_map.shape
    color_map = np.zeros((H, W, 3), dtype=np.uint8)
    for cls in range(palette.shape[0]):
        color_map[pred_map == cls] = palette[cls]
    return color_map


def save_classification_map(pred_map, dataset_name, H, W, visualize_dir="./visualization"):
    color_map = label_to_color(pred_map, dataset_name)
    os.makedirs(visualize_dir, exist_ok=True)
    save_path = os.path.join(visualize_dir, f"{dataset_name}_visualization.png")
    plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = plt.gca()
    ax.imshow(color_map)
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
    plt.margins(0, 0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    # print(f"分类可视化图已保存：{save_path}")
