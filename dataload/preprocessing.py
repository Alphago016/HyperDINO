import numpy as np
import os
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


def random_sampling(class_coords, num_samples, random_state=42):
    np.random.seed(random_state)
    if len(class_coords) <= num_samples:
        return class_coords
    sample_indices = np.random.choice(len(class_coords), num_samples, replace=False)
    sample_coords = [class_coords[idx] for idx in sample_indices]
    return sample_coords


def patch_coordinate_augmentation(coords, max_half, img_H, img_W, translate_range=2, random_state=42):
    np.random.seed(random_state)
    coords_arr = np.array(coords, dtype=np.int32)  # (N, 2)
    N = coords_arr.shape[0]

    # 批量生成随机偏移
    delta = np.random.randint(-translate_range, translate_range + 1, size=(N, 2), dtype=np.int32)
    new_coords = coords_arr + delta

    # 边界裁剪
    min_i = max_half
    max_i = img_H - 1 + max_half
    min_j = max_half
    max_j = img_W - 1 + max_half
    new_coords[:, 0] = np.clip(new_coords[:, 0], min_i, max_i)
    new_coords[:, 1] = np.clip(new_coords[:, 1], min_j, max_j)

    return list(map(tuple, new_coords))


def preprocess_image(image, labels, spectral_var_ratio=0.995, num_samples_per_class=20, random_state=42):

    H, W, C = image.shape
    flattened = image.reshape(-1, C)

    class_coords = {}
    all_labeled_coords = []
    for i in range(H):
        for j in range(W):
            cls = labels[i, j]
            if cls == 0:
                continue
            if cls not in class_coords:
                class_coords[cls] = []
            class_coords[cls].append((i, j))
            all_labeled_coords.append((i, j))

    train_coords = []
    train_set = set()
    for cls, coords in class_coords.items():
        sample_coords = random_sampling(
            class_coords=coords,
            num_samples=num_samples_per_class,
            random_state=random_state
        )
        train_coords.extend(sample_coords)
        train_set.update(sample_coords)

    test_coords = [coord for coord in all_labeled_coords if coord not in train_set]
    print(f"数据集固定划分完成：")
    print(f"  - 训练集：每类{num_samples_per_class}个，共{len(train_coords)}个样本")
    print(f"  - 测试集：共{len(test_coords)}个样本")

    train_spectral = np.array([image[i, j, :] for (i, j) in train_coords])
    scaler_spectral = StandardScaler()
    scaler_spectral.fit(train_spectral)
    flattened_scaled = scaler_spectral.transform(flattened)
    scaled_spectral_original = flattened_scaled.reshape(H, W, C)

    pca_dim = 64
    pca_dim = min(pca_dim, C, len(train_coords) - 1)
    spectral_pca = PCA(n_components=pca_dim, svd_solver='auto', random_state=random_state)
    spectral_pca.fit(scaler_spectral.transform(train_spectral))
    flattened_spectral_pca = spectral_pca.transform(flattened_scaled)
    actual_pca_dim = flattened_spectral_pca.shape[1]
    spectral_pca_feat = flattened_spectral_pca.reshape(H, W, actual_pca_dim)
    print(f"光谱PCA降维完成（训练集拟合，固定{actual_pca_dim}维去噪）：")
    print(f"  - 原始光谱维度：{C}")
    print(f"  - 累计方差贡献率：{np.sum(spectral_pca.explained_variance_ratio_):.4f}")

    spatial_input = scaled_spectral_original.astype(np.float32)
    print(f"空间分支输入构建完成：形状 {spatial_input.shape}（标准化全波段高光谱）")

    ch0 = scaled_spectral_original

    deriv1_full = np.gradient(ch0, axis=-1)
    train_deriv1 = np.array([deriv1_full[i, j, :] for (i, j) in train_coords])
    scaler_deriv1 = StandardScaler()
    scaler_deriv1.fit(train_deriv1)
    ch1 = scaler_deriv1.transform(deriv1_full.reshape(-1, C)).reshape(H, W, C)

    norms = np.linalg.norm(ch0, axis=-1, keepdims=True) + 1e-8
    ch2 = ch0 / norms
    train_norm = np.array([ch2[i, j, :] for (i, j) in train_coords])
    scaler_norm = StandardScaler()
    scaler_norm.fit(train_norm)
    ch2 = scaler_norm.transform(ch2.reshape(-1, C)).reshape(H, W, C)


    spectral_multiview_feat = np.stack([ch0, ch1, ch2], axis=2).astype(np.float32)
    print(f"多视图光谱特征构建完成（原始波长维度，3视图）：形状 {spectral_multiview_feat.shape}")
    print(f"  - Ch0: 标准化原始光谱（波长连续）")
    print(f"  - Ch1: 一阶导数特征（训练集标准化）")
    print(f"  - Ch2: L2归一化光谱形状（训练集标准化）")
    print(f"图像预处理完成：")
    print(f"  - 空间分支输入：{spatial_input.shape}")
    print(f"  - 光谱多视图输出：{spectral_multiview_feat.shape}")

    return (spatial_input, scaler_spectral, spectral_pca,
            spectral_pca_feat, scaled_spectral_original, spectral_multiview_feat,
            train_coords, test_coords)


def generate_patch_batch(image, labels, spatial_input, spectral_multiview_feat,
                         coords, patch_sizes, random_state=42, device="cuda", augment=True):

    np.random.seed(random_state)
    H, W, num_views, C = spectral_multiview_feat.shape
    patch_size = patch_sizes[0]
    half = patch_size // 2


    pad_width_spatial = ((half, half), (half, half), (0, 0))
    padded_spatial = np.pad(spatial_input, pad_width_spatial, mode='reflect')
    pad_width_spec = ((half, half), (half, half), (0, 0), (0, 0))
    padded_spectral = np.pad(spectral_multiview_feat, pad_width_spec, mode='reflect')


    coords_arr = np.array(coords, dtype=np.int32)
    N = coords_arr.shape[0]
    padded_coords_arr = coords_arr + half


    sample_labels = labels[coords_arr[:, 0], coords_arr[:, 1]].astype(int)
    sampled_labels = sample_labels - 1

    if augment:
        augmented_coords_arr = np.array(patch_coordinate_augmentation(
            coords=list(map(tuple, padded_coords_arr)),
            max_half=half,
            img_H=H,
            img_W=W,
            translate_range=2,
            random_state=random_state
        ), dtype=np.int32)
    else:
        augmented_coords_arr = padded_coords_arr

    rows = augmented_coords_arr[:, 0]
    cols = augmented_coords_arr[:, 1]

    row_off = np.arange(patch_size, dtype=np.int32) - half
    col_off = np.arange(patch_size, dtype=np.int32) - half
    patch_rows = rows[:, None, None] + row_off[None, :, None]
    patch_cols = cols[:, None, None] + col_off[None, None, :]
    patch_arr = padded_spatial[patch_rows, patch_cols, :]  # (N, patch_size, patch_size, C)

    spectral_arr = padded_spectral[rows, cols, :, :]

    if augment:

        flip_h = np.random.rand(N) < 0.5
        if flip_h.any():
            patch_arr[flip_h] = np.flip(patch_arr[flip_h], axis=2)

        flip_v = np.random.rand(N) < 0.5
        if flip_v.any():
            patch_arr[flip_v] = np.flip(patch_arr[flip_v], axis=1)

        rot_flag = np.random.rand(N) < 0.25
        if rot_flag.any():
            rot_idx = np.where(rot_flag)[0]
            k_vals = np.random.randint(1, 4, size=len(rot_idx))
            for k in [1, 2, 3]:
                k_mask = k_vals == k
                if k_mask.any():
                    idx = rot_idx[k_mask]
                    patch_arr[idx] = np.rot90(patch_arr[idx], k=k, axes=(1, 2))

        noise_flag = np.random.rand(N) < 0.5
        if noise_flag.any():
            noise_std = 0.02
            idx = np.where(noise_flag)[0]
            noise = np.random.normal(0, noise_std, patch_arr[idx].shape).astype(np.float32)
            patch_arr[idx] = np.clip(patch_arr[idx] + noise, -3.0, 3.0)

        scale_flag = np.random.rand(N) < 0.5
        if scale_flag.any():
            idx = np.where(scale_flag)[0]
            scales = np.random.uniform(0.95, 1.05, size=(len(idx), 1, 1)).astype(np.float32)
            spectral_arr[idx] *= scales

        shift_flag = np.random.rand(N) < 0.5
        if shift_flag.any():
            idx = np.where(shift_flag)[0]
            shifts = np.random.uniform(-0.02, 0.02, size=(len(idx), 1, 1)).astype(np.float32)
            spectral_arr[idx] += shifts

        spec_noise_flag = np.random.rand(N) < 0.5
        if spec_noise_flag.any():
            idx = np.where(spec_noise_flag)[0]
            noise = np.random.normal(0, 0.02, spectral_arr[idx].shape).astype(np.float32)
            spectral_arr[idx] += noise

        mask_flag = np.random.rand(N) < 0.5
        if mask_flag.any():
            idx = np.where(mask_flag)[0]
            num_mask = max(1, int(C * 0.05))
            for i in range(len(idx)):
                band_idx = np.random.choice(C, num_mask, replace=False)
                spectral_arr[idx[i], :, band_idx] = 0

    patch = torch.from_numpy(patch_arr).permute(0, 3, 1, 2).to(device)
    spectral_feat = torch.from_numpy(spectral_arr).to(device)
    labels_tensor = torch.from_numpy(sampled_labels).long().to(device)
    return patch, spectral_feat, labels_tensor


def batch_extract_features(dinov3_model, spatial_input, spectral_multiview_feat, labels,
                           coords, patch_sizes, batch_size=16,
                           save_path="./features/batch_features.npz", device="cuda", random_state=42):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    H, W, num_views, C = spectral_multiview_feat.shape
    max_patch_size = max(patch_sizes)
    max_half = max_patch_size // 2
    s1, s2 = patch_sizes
    half_s1 = s1 // 2
    half_s2 = s2 // 2

    pad_width_spatial = ((max_half, max_half), (max_half, max_half), (0, 0))
    padded_spatial = np.pad(spatial_input, pad_width_spatial, mode='reflect')
    pad_width_spec = ((max_half, max_half), (max_half, max_half), (0, 0), (0, 0))
    padded_spectral = np.pad(spectral_multiview_feat, pad_width_spec, mode='reflect')

    padded_coords = [(i + max_half, j + max_half) for (i, j) in coords]
    sampled_labels = [int(labels[i, j]) - 1 for (i, j) in coords]
    total_samples = len(padded_coords)
    print(f"预提取样本数：{total_samples}，视图数：{num_views}，波段数：{C}")

    all_feat_s1 = []
    all_feat_s2 = []
    all_spectral = []

    dinov3_model.eval()
    with torch.no_grad():
        for idx in tqdm(range(0, total_samples, batch_size), desc="批处理提取特征"):
            end_idx = min(idx + batch_size, total_samples)
            batch_coords = padded_coords[idx:end_idx]
            batch_patch_s1, batch_patch_s2, batch_spectral = [], [], []
            for (i, j) in batch_coords:
                batch_patch_s1.append(padded_spatial[i - half_s1:i + half_s1 + 1, j - half_s1:j + half_s1 + 1, :])
                batch_patch_s2.append(padded_spatial[i - half_s2:i + half_s2 + 1, j - half_s2:j + half_s2 + 1, :])
                batch_spectral.append(padded_spectral[i, j, :, :])
            patch_s1_tensor = torch.tensor(np.array(batch_patch_s1), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            patch_s2_tensor = torch.tensor(np.array(batch_patch_s2), dtype=torch.float32).permute(0, 3, 1, 2).to(device)

            from models.backbones import extract_dinov3_features
            feat_s1_np, _ = extract_dinov3_features(dinov3_model, patch_s1_tensor, device)
            feat_s2_np, _ = extract_dinov3_features(dinov3_model, patch_s2_tensor, device)
            feat_s1 = torch.from_numpy(feat_s1_np).to(device)
            feat_s2 = torch.from_numpy(feat_s2_np).to(device)

            all_feat_s1.extend(torch.mean(feat_s1, dim=(1, 2)).cpu().numpy())
            all_feat_s2.extend(torch.mean(feat_s2, dim=(1, 2)).cpu().numpy())
            all_spectral.extend(batch_spectral)

    np.savez(save_path,
             feat_s1=np.array(all_feat_s1),
             feat_s2=np.array(all_feat_s2),
             spectral=np.array(all_spectral),
             labels=np.array(sampled_labels))
    print(f"特征已保存至：{save_path}")
    return save_path


def sliding_window_dense_feature(model, img_np, patch_size, overlap=0, device="cuda"):

    from models.backbones import extract_dinov3_features
    H, W, C = img_np.shape
    half = patch_size // 2
    pad = half
    img_pad = np.pad(img_np, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    dense_feat = np.zeros((H, W, 1024), dtype=np.float32)

    model.eval()
    batch_size = 64
    coords = []
    for y0 in range(H):
        for x0 in range(W):
            coords.append((y0, x0))

    with torch.no_grad():
        for i in tqdm(range(0, len(coords), batch_size), desc=f"Dense inference patch_size={patch_size}"):
            batch_coords = coords[i:i+batch_size]
            batch_crops = []
            for (y0, x0) in batch_coords:
                crop = img_pad[y0:y0 + patch_size, x0:x0 + patch_size, :]
                batch_crops.append(crop)
            crop_t = torch.from_numpy(np.array(batch_crops)).permute(0, 3, 1, 2).float().to(device)
            feat_b, _ = extract_dinov3_features(model, crop_t, device)
            feat_vec = feat_b.mean(axis=(1, 2))
            for idx, (y0, x0) in enumerate(batch_coords):
                dense_feat[y0, x0, :] = feat_vec[idx]
    return dense_feat
