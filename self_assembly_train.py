# train.py
import os
import torch
from torch.utils.data import DataLoader, random_split
import argparse
from tqdm import tqdm
from dataset import AssemblyDataset, collate_fn
from models import PairClassifier
from utils import set_seed, compute_metrics, save_model
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

torch.cuda.empty_cache()


# ---- train_epoch ----
def train_epoch(model, loader, opt, device):
    model.train()
    losses = []
    all_probs = []
    all_labels = []
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for batch in tqdm(loader, desc="train"):
        # 优化：更简洁地将 batch 内所有 Tensor 移至 GPU
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)

        opt.zero_grad()
        logits = model(batch)
        labels = batch['labels'].to(device).float()

        loss = loss_fn(logits, labels)
        loss.backward()

        # 【新增】：梯度裁剪，防止 Attention 机制导致梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        opt.step()
        losses.append(loss.item())

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    return sum(losses) / len(losses), metrics


# ---- eval_epoch ----
def eval_epoch(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    val_losses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)

            logits = model(batch)
            labels = batch['labels'].to(device).float()

            loss = loss_fn(logits, labels)
            val_losses.append(loss.item())

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    val_loss = sum(val_losses) / len(val_losses)
    return val_loss, metrics


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # load dataset
    ds = AssemblyDataset(args.processed)
    n = len(ds)
    val_n = int(n * args.val_ratio)
    train_n = n - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    sample0 = ds[0]
    node_feat_dim = sample0['a_atom_feats'].shape[1]
    fp_dim = sample0['a_fp'].shape[0]

    # 【修改】：移除了 use_layernorm 参数，对应新模型的初始化
    model = PairClassifier(
        node_feat_dim=node_feat_dim,
        fp_dim=fp_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_gcn_layers,
        rbf_dim=args.rbf_dim,
        dropout=0.1
    )
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  # 将 weight_decay 设为 1e-4 更适合复杂模型

    # 【新增】：学习率调度器，监控验证集 AUC，如果 5 个 epoch 不提升则学习率减半
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, verbose=True)

    best_auc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, opt, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch} | LR: {opt.param_groups[0]['lr']:.2e} | "
              f"Train Loss: {train_loss:.4f} | Train AUC: {train_metrics['auc']:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {val_metrics['auc']:.4f} | Val Acc: {val_metrics['acc']:.4f}")

        # 更新学习率调度器
        scheduler.step(val_metrics['auc'])

        # save best by AUC
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_path = os.path.join(args.save_dir, f"best_model_epoch{epoch}_auc{best_auc:.4f}.pt")
            save_model(model, save_path)
            print(">>> Saved new best model to", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="out processed_data.pt")
    parser.add_argument("--batch_size", type=int, default=8)  # 如果显存够，建议设为 16 或 32
    parser.add_argument("--epochs", type=int, default=65)
    # 【建议】：加入 Attention 后，初始学习率可以适当提高一点，比如 5e-5 或 1e-4
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_gcn_layers", type=int, default=4)  # 建议先从 4 层跑起，Attention 层多容易过平滑
    parser.add_argument("--rbf_dim", type=int, default=32)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()
    main(args)