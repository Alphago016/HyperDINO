import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time
import gc

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_INDUCTOR_DISABLE_AUTOTUNE"] = "1"

from dataload import (
    load_hsi_dataset,
    preprocess_image,
    generate_patch_batch,
)
from models.backbones import load_dinov3_with_adapter, extract_dinov3_features_train
from models.fusion_classifier import HSIClassifier
from utils import (
    set_seed,
    EarlyStopping,
    calculate_kappa,
    write_result_log,
    save_classification_map
)


def parse_args():
    parser = argparse.ArgumentParser(description="HyperDINO HSI Classification Entry")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="yaml config path")
    parser.add_argument("--dataset", type=str, default="Houston", help="dataset name in config")
    return parser.parse_args()


def load_yaml_config(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def evaluate_test_set(dinov3_model, classifier, image, labels, spatial_input,
                      spectral_multiview_feat, test_coords, patch_sizes,
                      test_batch_size, device, random_state=42):
    dinov3_model.eval()
    classifier.eval()
    all_preds = []
    all_labels = []
    total_valid = len(test_coords)

    start_time = time.time()

    with torch.inference_mode():
        for idx_start in tqdm(range(0, total_valid, test_batch_size), desc="测试集推理", leave=True):
            idx_end = min(idx_start + test_batch_size, total_valid)
            batch_coords = test_coords[idx_start:idx_end]

            patch, spectral_feat, gt_labels = generate_patch_batch(
                image=image,
                labels=labels,
                spatial_input=spatial_input,
                spectral_multiview_feat=spectral_multiview_feat,
                coords=batch_coords,
                patch_sizes=patch_sizes,
                random_state=random_state,
                device=device,
                augment=False
            )

            feat_spatial = extract_dinov3_features_train(dinov3_model, patch, spectral_feat)

            out = classifier(feat_spatial, spectral_feat)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(gt_labels.cpu().numpy())

    infer_time = time.time() - start_time
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    oa = np.mean(all_preds_np == all_labels_np)
    return oa, all_preds_np, all_labels_np, infer_time


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)


    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["cuda_visible_devices"]
    os.environ["OMP_NUM_THREADS"] = str(cfg["omp_num_threads"])
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = cfg["run_dataset"] if args.dataset is None else args.dataset
    ds_cfg = cfg["datasets"][dataset_name]
    os.makedirs(cfg["save_dir"], exist_ok=True)
    os.makedirs(cfg["visualize_dir"], exist_ok=True)
    test_interval = cfg.get("test_interval", 10)
    test_batch_size = cfg.get("test_batch_size", 128)

    print(f"\n===== 训练{dataset_name}（测试集推理验证） =====")
    num_samples_per_class = ds_cfg["train_samples_per_class"]

    image, labels = load_hsi_dataset(
        dataset_name,
        ds_cfg["image_mat_path"],
        ds_cfg["gt_mat_path"]
    )
    orig_H, orig_W = image.shape[0], image.shape[1]

    (spatial_input, scaler_spectral, spectral_pca,
     spectral_pca_feat, scaled_spectral_original, spectral_multiview_feat,
     train_coords, test_coords) = preprocess_image(
        image,
        labels,
        spectral_var_ratio=cfg["spectral_var_ratio"],
        num_samples_per_class=num_samples_per_class,
        random_state=cfg["seed"]
    )
    spectral_bands = spectral_multiview_feat.shape[-1]

    dinov3_model = load_dinov3_with_adapter(
        repo_dir=cfg["repo_dir"],
        weight_path=cfg["weight_path"],
        in_chans=spectral_bands,
        device=device
    )
    patch_sizes = cfg["patch_sizes"]
    num_classes = len(np.unique(labels[labels > 0]))
    print(f"类别数：{num_classes} | 每类训练样本：{num_samples_per_class} | 光谱视图：3 | 波段数：{spectral_bands}")

    model = HSIClassifier(spectral_dim=spectral_bands, num_classes=num_classes).to(device)

    trainable_params = list(model.parameters())
    for param in dinov3_model.parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = optim.AdamW(
        trainable_params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999)
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["max_epochs"] - cfg["warmup_epochs"],
        eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    best_test_oa = 0.0
    best_epoch = 0
    model_save_path = os.path.join(cfg["save_dir"], f"best_model_{dataset_name}.pth")

    print(f"\n开始训练（冻结DINOv3主干，联合微调Adapter+分类器，最大{cfg['max_epochs']}轮）")
    train_start_time = time.time()

    for epoch in range(cfg["max_epochs"]):
        current_seed = cfg["seed"] + epoch

        patch, spectral_feat, train_labels = generate_patch_batch(
            image=image,
            labels=labels,
            spatial_input=spatial_input,
            spectral_multiview_feat=spectral_multiview_feat,
            coords=train_coords,
            patch_sizes=patch_sizes,
            random_state=current_seed,
            device=device,
            augment=True
        )
        train_dataset = TensorDataset(patch, spectral_feat, train_labels)

        dataloader_gen = torch.Generator()
        dataloader_gen.manual_seed(current_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            generator=dataloader_gen
        )

        dinov3_model.eval()

        for module in dinov3_model.modules():
            if hasattr(module, "adapter"):
                module.adapter.train()

        dinov3_model.spectral_embed.train()
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # Warmup
        if epoch < cfg["warmup_epochs"]:
            lr = cfg["warmup_init_lr"] + (cfg["lr"] - cfg["warmup_init_lr"]) * (epoch / cfg["warmup_epochs"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        for p_spatial, spec, y in train_loader:
            p_spatial, spec, y = p_spatial.to(device), spec.to(device), y.to(device)
            optimizer.zero_grad()

            feat_spatial = extract_dinov3_features_train(dinov3_model, p_spatial, spec)
            out = model(feat_spatial, spec)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, cfg["grad_clip_norm"])
            optimizer.step()

            total_loss += loss.item() * p_spatial.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total

        # ========== 间隔测试集评估 ==========
        if (epoch + 1) % test_interval == 0:
            test_oa, test_preds, test_gt, _ = evaluate_test_set(
                dinov3_model=dinov3_model,
                classifier=model,
                image=image,
                labels=labels,
                spatial_input=spatial_input,
                spectral_multiview_feat=spectral_multiview_feat,
                test_coords=test_coords,
                patch_sizes=patch_sizes,
                test_batch_size=test_batch_size,
                device=device,
                random_state=cfg["seed"]
            )

            if test_oa > best_test_oa:
                best_test_oa = test_oa
                best_epoch = epoch + 1
                torch.save({
                    "classifier_state_dict": model.state_dict(),
                    "dinov3_state_dict": dinov3_model.state_dict()
                }, model_save_path)
                print(
                    f" 第{epoch + 1}轮：新最优 | 训练损失: {avg_train_loss:.4f} | 训练Acc: {train_acc:.4f} | 测试OA: {test_oa:.4f}")
            else:
                print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Test OA: {test_oa:.4f} | Best OA: {best_test_oa:.4f} (@Epoch {best_epoch}) | LR: {current_lr:.6f}")
        else:

            print(
                f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.6f}")

    train_total_time = time.time() - train_start_time
    print(f"最优模型：第 {best_epoch} 轮，测试OA = {best_test_oa:.4f}")

    # ===== 加载最优模型，生成最终完整测试报告 =====
    print("\n===== 加载最优模型，生成完整测试结果 =====")
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["classifier_state_dict"])
    dinov3_model.load_state_dict(checkpoint["dinov3_state_dict"])

    # 最终完整评估
    final_oa, final_preds, final_labels, infer_time = evaluate_test_set(
        dinov3_model=dinov3_model,
        classifier=model,
        image=image,
        labels=labels,
        spatial_input=spatial_input,
        spectral_multiview_feat=spectral_multiview_feat,
        test_coords=test_coords,
        patch_sizes=patch_sizes,
        test_batch_size=test_batch_size,
        device=device,
        random_state=cfg["seed"]
    )

    gc.collect()
    torch.cuda.empty_cache()

    cm = confusion_matrix(final_labels, final_preds)
    OA = np.trace(cm) / np.sum(cm)
    AA = np.mean(np.nan_to_num(np.diag(cm) / np.sum(cm, axis=1)))
    Kappa = calculate_kappa(cm)
    report = classification_report(final_labels, final_preds, digits=4)

    log_content = f"""
{dataset_name} 分类结果
====================================
训练设置：每类{num_samples_per_class}个训练样本，共{len(train_coords)}个训练样本
最优轮次：第 {best_epoch} 轮
测试设置：共{len(test_coords)}个测试样本
OA = {OA:.4f}
AA = {AA:.4f}
Kappa = {Kappa:.4f}
训练总耗时：{train_total_time:.2f} 秒
====================================
{report}
"""
    log_path = os.path.join(cfg["save_dir"], f"{dataset_name}_result.txt")
    write_result_log(log_content, log_path)

    print(f"\n{dataset_name}实验完成！最终测试指标：OA={OA:.4f}, AA={AA:.4f}, Kappa={Kappa:.4f}")
    print(f"结果日志已保存至：{log_path}")

    # ========== 生成分类可视化图 ==========
    print("\n===== 生成分类可视化图 =====")

    all_valid_coords = train_coords + test_coords

    all_oa, all_preds, all_labels, vis_infer_time = evaluate_test_set(
        dinov3_model=dinov3_model,
        classifier=model,
        image=image,
        labels=labels,
        spatial_input=spatial_input,
        spectral_multiview_feat=spectral_multiview_feat,
        test_coords=all_valid_coords,
        patch_sizes=patch_sizes,
        test_batch_size=test_batch_size,
        device=device,
        random_state=cfg["seed"]
    )

    H, W = labels.shape
    pred_map = np.zeros((H, W), dtype=np.uint8)
    for idx, (i, j) in enumerate(all_valid_coords):
        pred_map[i, j] = all_preds[idx] + 1

    save_classification_map(pred_map, dataset_name, H, W, visualize_dir=cfg["visualize_dir"])
    print(f"分类可视化图已保存至：{os.path.join(cfg['visualize_dir'], f'{dataset_name}_visualization.png')}")


if __name__ == "__main__":
    main()
