
import os
import re
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

@torch.no_grad()
def sample_theta_mean(model, batch, device, steps=100, n_samples=5):
    preds = []

    for _ in range(n_samples):
        pred = sample_theta(model, batch, device, steps=steps)
        preds.append(pred.cpu().numpy())

    pred_mean = np.mean(preds, axis=0)
    return torch.tensor(pred_mean, dtype=torch.float32, device=device)


def classify_kappa(kappa, thresholds=None):
    """
    将 kappa 转成协同类别。
    thresholds 可以根据你的训练集分布或领域定义调整。

    这里给一个示例：
    class 0: antagonistic / low
    class 1: weak
    class 2: moderate
    class 3: strong
    """
    if thresholds is None:
        thresholds = [-1.0, 1.0, 5.0]

    kappa = np.asarray(kappa)

    labels = np.zeros_like(kappa, dtype=int)
    labels[kappa >= thresholds[0]] = 1
    labels[kappa >= thresholds[1]] = 2
    labels[kappa >= thresholds[2]] = 3

    return labels


def compute_core_ablation_metrics(pred_norm, true_norm, pred_log, true_log, pred_phys, true_phys):
    """
    计算消融实验最核心的指标。
    pred_norm / true_norm: 标准化空间
    pred_log / true_log: E0, Ef, logEC50A, logEC50B, mA, mB, kappa
    pred_phys / true_phys: EC50 已还原到物理浓度
    """

    param_names = ["E0", "Ef", "logEC50A", "logEC50B", "mA", "mB", "kappa"]
    kappa_idx = 6

    rows = []

    # 1. 每个参数的 R2 / MAE / RMSE，建议主要看 log_space
    for i, name in enumerate(param_names):
        y_true = true_log[:, i]
        y_pred = pred_log[:, i]

        rows.append({
            "metric_group": "parameter_log_space",
            "metric": f"{name}_R2",
            "value": r2_score(y_true, y_pred)
        })
        rows.append({
            "metric_group": "parameter_log_space",
            "metric": f"{name}_MAE",
            "value": mean_absolute_error(y_true, y_pred)
        })
        rows.append({
            "metric_group": "parameter_log_space",
            "metric": f"{name}_RMSE",
            "value": rmse_score(y_true, y_pred)
        })

    # 2. kappa 重点指标
    y_true_k = true_log[:, kappa_idx]
    y_pred_k = pred_log[:, kappa_idx]

    rows.extend([
        {
            "metric_group": "kappa_regression",
            "metric": "kappa_R2",
            "value": r2_score(y_true_k, y_pred_k)
        },
        {
            "metric_group": "kappa_regression",
            "metric": "kappa_MAE",
            "value": mean_absolute_error(y_true_k, y_pred_k)
        },
        {
            "metric_group": "kappa_regression",
            "metric": "kappa_RMSE",
            "value": rmse_score(y_true_k, y_pred_k)
        }
    ])

    # 3. BRAID-zMAE：在标准化空间平均 7 个参数误差
    # 这个适合比较不同模型整体 BRAID 参数预测能力
    braid_zmae = np.mean(np.abs(pred_norm - true_norm))

    rows.append({
        "metric_group": "braid_overall",
        "metric": "BRAID_zMAE",
        "value": braid_zmae
    })

    # 4. EC50-logMAE：只看两个 logEC50 参数
    ec50_log_mae = np.mean(
        np.abs(pred_log[:, [2, 3]] - true_log[:, [2, 3]])
    )

    rows.append({
        "metric_group": "dose_sensitivity",
        "metric": "EC50_logMAE",
        "value": ec50_log_mae
    })

    # 5. slope-MAE：mA 和 mB 的平均误差
    m_mae = np.mean(
        np.abs(pred_log[:, [4, 5]] - true_log[:, [4, 5]])
    )

    rows.append({
        "metric_group": "dose_slope",
        "metric": "m_MAE",
        "value": m_mae
    })

    # 6. 协同状态分类指标
    true_cls = classify_kappa(y_true_k)
    pred_cls = classify_kappa(y_pred_k)

    rows.extend([
        {
            "metric_group": "synergy_classification",
            "metric": "Accuracy",
            "value": accuracy_score(true_cls, pred_cls)
        },
        {
            "metric_group": "synergy_classification",
            "metric": "Macro_F1",
            "value": f1_score(true_cls, pred_cls, average="macro", zero_division=0)
        },
        {
            "metric_group": "synergy_classification",
            "metric": "Macro_Precision",
            "value": precision_score(true_cls, pred_cls, average="macro", zero_division=0)
        },
        {
            "metric_group": "synergy_classification",
            "metric": "Macro_Recall",
            "value": recall_score(true_cls, pred_cls, average="macro", zero_division=0)
        }
    ])

    # 7. 二分类 AUROC / AUPRC：例如 strong synergy vs others
    # 这里把 kappa >= 5 视为 strong synergy，可按你的定义调整
    y_true_bin = (y_true_k >= 5.0).astype(int)
    y_score = y_pred_k

    if len(np.unique(y_true_bin)) > 1:
        rows.append({
            "metric_group": "strong_synergy_detection",
            "metric": "AUROC",
            "value": roc_auc_score(y_true_bin, y_score)
        })
        rows.append({
            "metric_group": "strong_synergy_detection",
            "metric": "AUPRC",
            "value": average_precision_score(y_true_bin, y_score)
        })
    else:
        rows.append({
            "metric_group": "strong_synergy_detection",
            "metric": "AUROC",
            "value": np.nan
        })
        rows.append({
            "metric_group": "strong_synergy_detection",
            "metric": "AUPRC",
            "value": np.nan
        })

    return pd.DataFrame(rows)

# ============================================================
# 0. 配置
# ============================================================

CSV_PATH = "braid_results_final_with_smiles1.csv"
SAVE_DIR = "flow_result"

MAX_LEN = 128
BATCH_SIZE = 128
EPOCHS = 100
LR = 2e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 0.5

VAL_RATIO = 0.15
RANDOM_SEED = 42

DROP_DUPLICATE_BLOCKID = True
FILTER_FIT_SUCCESS = True

# 采样步数。越大越稳定，但更慢
EVAL_STEPS = 100
EVAL_SAMPLES = 20

# 是否在训练时随机交换 A/B
AUGMENT_SWAP_AB = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. 随机种子
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. SMILES tokenizer
# ============================================================

SMILES_PATTERN = re.compile(
    r"(\[[^\]]+\]"          # bracket atoms, e.g. [nH], [C@@H], [Na+]
    r"|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|Zn|Fe|Cu|Mn|Co|Ni"
    r"|@@?"
    r"|%\d{2}"              # two-digit ring index, e.g. %12
    r"|\d"                  # one-digit ring index
    r"|=|#|-|\+|\\|/|:|~|\."
    r"|\(|\)"
    r"|\*|\$"
    r"|[BCNOFPSIbcnops])"
)


def smiles_tokenizer(smiles, strict=True):
    """
    稳健 SMILES tokenizer。
    strict=True 时，遇到无法识别的字符会报错，避免静默丢信息。
    """
    if pd.isna(smiles):
        return []

    smiles = str(smiles).strip()

    tokens = []
    pos = 0

    for match in SMILES_PATTERN.finditer(smiles):
        start, end = match.span()

        if start != pos:
            unknown = smiles[pos:start]
            if strict:
                raise ValueError(
                    f"Unrecognized SMILES token: {unknown!r} in {smiles!r}"
                )

        tokens.append(match.group())
        pos = end

    if pos != len(smiles):
        unknown = smiles[pos:]
        if strict:
            raise ValueError(
                f"Unrecognized SMILES token at end: {unknown!r} in {smiles!r}"
            )

    return tokens


def build_vocab(smiles_list):
    vocab = {"PAD": 0, "UNK": 1}
    idx = 2

    for smi in smiles_list:
        tokens = smiles_tokenizer(smi, strict=True)
        for t in tokens:
            if t not in vocab:
                vocab[t] = idx
                idx += 1

    return vocab


def check_smiles_tokenization(df):
    bad_rows = []

    for col in ["SMILES_A", "SMILES_B"]:
        for i, smi in enumerate(df[col]):
            try:
                _ = smiles_tokenizer(smi, strict=True)
            except Exception as e:
                bad_rows.append({
                    "row": i,
                    "column": col,
                    "smiles": smi,
                    "error": str(e)
                })

    bad_df = pd.DataFrame(bad_rows)

    if len(bad_df) > 0:
        bad_df.to_csv("bad_smiles_tokenization.csv", index=False)
        raise ValueError(
            f"发现 {len(bad_df)} 条 SMILES 无法 tokenize，"
            f"已保存 bad_smiles_tokenization.csv，请先修正。"
        )

    print("SMILES tokenization 检查通过。")
    return bad_df


# ============================================================
# 3. 数据预处理
# ============================================================

def load_and_clean_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required_cols = [
        "fit_success",
        "SMILES_A", "SMILES_B", "score",
        "E0", "Ef", "EC50A", "EC50B", "mA", "mB", "kappa"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列: {missing}")

    if FILTER_FIT_SUCCESS:
        fit_col = df["fit_success"]
        if fit_col.dtype == bool:
            df = df[fit_col].copy()
        else:
            df = df[fit_col.astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    # 清理空字符串
    for col in ["SMILES_A", "SMILES_B"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    df = df.dropna(subset=["SMILES_A", "SMILES_B", "score"]).copy()

    if DROP_DUPLICATE_BLOCKID and "BlockID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["BlockID"], keep="first").copy()
        after = len(df)
        if before != after:
            print(f"按 BlockID 去重: {before} -> {after}")

    numeric_cols = ["score", "E0", "Ef", "EC50A", "EC50B", "mA", "mB", "kappa"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=numeric_cols).copy()

    # EC50 必须为正
    df = df[(df["EC50A"] > 0) & (df["EC50B"] > 0)].copy()

    df = df.reset_index(drop=True)
    return df


# ============================================================
# 4. Dataset
# ============================================================

class FlowDataset(Dataset):
    def __init__(self, df, vocab, max_len=128, train=True, mean=None, std=None, cell2id=None):
        self.df = df.copy().reset_index(drop=True)
        # 训练目标：EC50 使用 log10
        self.param_cols = ["E0", "Ef", "EC50A", "EC50B", "mA", "mB", "kappa"]

        theta_df = self.df[self.param_cols].copy()
        theta_df["EC50A"] = np.log10(theta_df["EC50A"].values + 1e-6)
        theta_df["EC50B"] = np.log10(theta_df["EC50B"].values + 1e-6)

        theta = theta_df.values.astype(np.float32)

        if mean is None:
            self.mean = theta.mean(axis=0).astype(np.float32)
        else:
            self.mean = np.asarray(mean, dtype=np.float32)

        if std is None:
            self.std = (theta.std(axis=0) + 1e-6).astype(np.float32)
        else:
            self.std = np.asarray(std, dtype=np.float32)

        self.theta = ((theta - self.mean) / self.std).astype(np.float32)
        self.theta_min = theta.min(axis=0).astype(np.float32)
        self.theta_max = theta.max(axis=0).astype(np.float32)

        self.smiles_a = self.df["SMILES_A"].values
        self.smiles_b = self.df["SMILES_B"].values
        self.score = self.df["score"].values.astype(np.float32)

        if "Cell Line" not in self.df.columns:
            raise ValueError("CSV missing required column: Cell Line")

        if cell2id is None:
            self.cell2id = {"UNK_CELL": 0}
            for c in self.df["Cell Line"].astype(str).unique():
                if c not in self.cell2id:
                    self.cell2id[c] = len(self.cell2id)
        else:
            self.cell2id = dict(cell2id)

        self.cells = self.df["Cell Line"].astype(str).map(
            lambda x: self.cell2id.get(x, self.cell2id["UNK_CELL"])
        ).values.astype(np.int64)

        self.vocab = vocab
        self.max_len = max_len
        self.train = train

    def encode(self, smiles):
        tokens = smiles_tokenizer(smiles, strict=True)
        ids = [self.vocab.get(t, 1) for t in tokens]
        ids = ids[:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def __getitem__(self, idx):
        a_ids = self.encode(self.smiles_a[idx])
        b_ids = self.encode(self.smiles_b[idx])
        theta = self.theta[idx].copy()
        score = self.score[idx]

        # 随机交换 A/B，保持药物组合对称性
        if self.train and AUGMENT_SWAP_AB and np.random.rand() > 0.5:
            a_ids, b_ids = b_ids, a_ids
            # theta 索引：
            # 0 E0, 1 Ef, 2 EC50A, 3 EC50B, 4 mA, 5 mB, 6 kappa
            theta[2], theta[3] = theta[3], theta[2]
            theta[4], theta[5] = theta[5], theta[4]

        return {
            "a": torch.tensor(a_ids, dtype=torch.long),
            "b": torch.tensor(b_ids, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.float32),
            "cell": torch.tensor(self.cells[idx]),
            "theta": torch.tensor(theta, dtype=torch.float32)
        }

    def __len__(self):
        return len(self.theta)


# ============================================================
# 5. 模型
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb_scale = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size, dim=128, nhead=4, layers=4):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            batch_first=True,
            dropout=0.1,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, layers)

    def forward(self, x):
        # True 表示 PAD
        pad_mask = x.eq(0)

        h = self.emb(x)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        return h, pad_mask


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.scale = dim ** -0.5

    def forward(self, A, B, key_padding_mask=None):
        """
        A attends to B.

        A: [B, LA, D]
        B: [B, LB, D]
        key_padding_mask: [B, LB], True 表示 B 中该位置是 PAD
        """
        Q = self.q(A)
        K = self.k(B)
        V = self.v(B)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1), -1e9)

        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        return out


def masked_mean(x, pad_mask):
    """
    x: [B, L, D]
    pad_mask: [B, L], True 表示 PAD
    """
    valid = (~pad_mask).float().unsqueeze(-1)
    x = x * valid
    denom = valid.sum(dim=1).clamp(min=1.0)
    return x.sum(dim=1) / denom


class PairEncoder(nn.Module):
    def __init__(self, vocab_size, dim=128):
        super().__init__()

        self.encoder = SmilesEncoder(vocab_size, dim)

        self.cross_ab = CrossAttention(dim)
        self.cross_ba = CrossAttention(dim)

        self.fc = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU()
        )

    def forward(self, a, b):
        A, maskA = self.encoder(a)
        B, maskB = self.encoder(b)

        A2B = self.cross_ab(A, B, key_padding_mask=maskB)
        B2A = self.cross_ba(B, A, key_padding_mask=maskA)

        a_vec = masked_mean(A, maskA)
        b_vec = masked_mean(B, maskB)
        a2b_vec = masked_mean(A2B, maskA)
        b2a_vec = masked_mean(B2A, maskB)

        pair = torch.cat([a_vec, b_vec, a2b_vec, b2a_vec], dim=-1)

        return self.fc(pair)


class FlowModel(nn.Module):
    def __init__(self, vocab_size,num_cells):
        super().__init__()

        # 药物对编码
        self.pair_encoder = PairEncoder(vocab_size, dim=128)
        self.cell_emb = nn.Embedding(num_cells, 32)
        # score 编码
        self.score_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 128),
            nn.SiLU()
        )

        # score gate，保留你原来的设计
        self.score_gate = nn.Sequential(
            nn.Linear(1, 128),
            nn.Sigmoid()
        )

        # 条件融合：pair 128 + score_feat 128 -> cond 256
        self.cond_mlp = nn.Sequential(
            nn.Linear(128 + 32 + 128, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU()
        )

        # 关键：theta_t 是 7 维状态，但进入 MLP 前升维到 128
        # 这不是旧版的 latent encoder，不需要 decoder。
        self.theta_in = nn.Sequential(
            nn.Linear(7, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU()
        )

        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(64),
            nn.Linear(64, 128),
            nn.SiLU()
        )

        # Flow 网络：
        # theta_feat 128 + t_emb 128 + cond 256 + score_feat 128 = 640
        # 输出必须是 7 维速度
        self.flow = nn.Sequential(
            nn.Linear(128 + 128 + 256 + 128, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 7)
        )

    def forward(self, batch):
        pair_emb = self.pair_encoder(batch["a"], batch["b"])  # [B, 128]
        cell_emb = self.cell_emb(batch["cell"])  # [B,32]
        score = batch["score"].unsqueeze(-1)                  # [B, 1]

        # score gate
        gate = self.score_gate(score)
        pair_emb = pair_emb * gate

        # score feature
        score_feat = self.score_mlp(score)                    # [B, 128]

        cond_input = torch.cat([pair_emb, cell_emb, score_feat], dim=-1)
        cond = self.cond_mlp(cond_input)                      # [B, 256]

        return cond, score_feat

    def flow_forward(self, theta_t, t, cond, score_feat):
        """
        theta_t: [B, 7]
        t:       [B, 1]
        cond:    [B, 256]
        score_feat: [B, 128]

        return:
            v: [B, 7]
        """
        theta_feat = self.theta_in(theta_t)                   # [B, 128]
        t_emb = self.time_mlp(t)                              # [B, 128]

        x = torch.cat([theta_feat, t_emb, cond, score_feat], dim=-1)
        v = self.flow(x)

        return v


# ============================================================
# 6. Flow Matching loss：直接 7 维 theta 空间
# ============================================================

def flow_loss(model, batch, device):
    theta = batch["theta"].to(device)     # [B, 7]

    cond, score_feat = model(batch)

    B = theta.size(0)

    z = torch.randn_like(theta)           # [B, 7]
    t = torch.rand(B, 1).to(device)

    theta_t = t * theta + (1.0 - t) * z
    v_target = theta - z                  # [B, 7]

    v_pred = model.flow_forward(theta_t, t, cond, score_feat)

    loss = F.mse_loss(v_pred, v_target)

    return loss


# ============================================================
# 7. 采样与评估
# ============================================================

@torch.no_grad()
def sample_theta(model, batch, device, steps=100):
    """
    从标准正态噪声积分到标准化 theta。
    输出 theta_norm: [B, 7]
    """
    model.eval()

    B = batch["a"].size(0)
    theta_t = torch.randn(B, 7).to(device)

    cond, score_feat = model(batch)

    dt = 1.0 / steps

    for i in range(steps):
        t_val = i / steps
        t = torch.ones(B, 1, device=device) * t_val

        v_pred = model.flow_forward(theta_t, t, cond, score_feat)
        theta_t = theta_t + v_pred * dt

    return theta_t


def inverse_transform_theta(theta_norm, mean, std):
    """
    theta_norm: numpy array [N, 7]
    mean/std: numpy array [7]

    返回物理尺度：
        EC50A/EC50B 会从 log10 空间还原到原始 uM。
    """
    theta_phys = theta_norm * std + mean

    # 还原 EC50
    theta_phys[:, 2] = np.power(10.0, theta_phys[:, 2]) - 1e-6
    theta_phys[:, 3] = np.power(10.0, theta_phys[:, 3]) - 1e-6

    # 避免极小数值误差导致负 EC50
    theta_phys[:, 2] = np.clip(theta_phys[:, 2], 0.0, None)
    theta_phys[:, 3] = np.clip(theta_phys[:, 3], 0.0, None)

    return theta_phys


@torch.no_grad()
def evaluate_model(model, dataloader, dataset_for_stats, device, save_dir, steps=100):
    """
    同时输出：
    1. 标准化空间 metrics
    2. log 空间 metrics，EC50 仍为 log10
    3. 物理空间 metrics，EC50 还原为 uM
    """
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    all_pred_norm = []
    all_true_norm = []

    for batch in dataloader:
        for k in batch:
            batch[k] = batch[k].to(device)

        pred_norm = sample_theta_mean(
            model,
            batch,
            device,
            steps=steps,
            n_samples=EVAL_SAMPLES
        )
        true_norm = batch["theta"]

        all_pred_norm.append(pred_norm.cpu().numpy())
        all_true_norm.append(true_norm.cpu().numpy())

    pred_norm = np.concatenate(all_pred_norm, axis=0)
    true_norm = np.concatenate(all_true_norm, axis=0)

    mean = dataset_for_stats.mean
    std = dataset_for_stats.std

    pred_log = pred_norm * std + mean
    true_log = true_norm * std + mean

    if hasattr(dataset_for_stats, "theta_min") and hasattr(dataset_for_stats, "theta_max"):
        pred_log = np.clip(pred_log, dataset_for_stats.theta_min, dataset_for_stats.theta_max)
        pred_norm = ((pred_log - mean) / std).astype(np.float32)
        true_norm = ((true_log - mean) / std).astype(np.float32)

    pred_phys = inverse_transform_theta(pred_norm, mean, std)
    true_phys = inverse_transform_theta(true_norm, mean, std)

    advanced_metrics = compute_core_ablation_metrics(
        pred_norm=pred_norm,
        true_norm=true_norm,
        pred_log=pred_log,
        true_log=true_log,
        pred_phys=pred_phys,
        true_phys=true_phys
    )

    advanced_metrics.to_csv(
        os.path.join(save_dir, "advanced_ablation_metrics.csv"),
        index=False
    )

    print("\n=== Advanced Ablation Metrics ===")
    print(advanced_metrics)


    param_names = ["E0", "Ef", "EC50A", "EC50B", "mA", "mB", "kappa"]

    rows_norm = []
    rows_log = []
    rows_phys = []

    for i, name in enumerate(param_names):
        for rows, pred, true, space in [
            (rows_norm, pred_norm, true_norm, "zscore"),
            (rows_log, pred_log, true_log, "log_space"),
            (rows_phys, pred_phys, true_phys, "physical")
        ]:
            try:
                r2 = r2_score(true[:, i], pred[:, i])
            except Exception:
                r2 = np.nan

            mae = mean_absolute_error(true[:, i], pred[:, i])

            rows.append({
                "space": space,
                "Parameter": name,
                "R2": r2,
                "MAE": mae,
                "GT_mean": float(np.mean(true[:, i])),
                "Pred_mean": float(np.mean(pred[:, i])),
                "GT_std": float(np.std(true[:, i])),
                "Pred_std": float(np.std(pred[:, i])),
            })

    metrics_norm = pd.DataFrame(rows_norm)
    metrics_log = pd.DataFrame(rows_log)
    metrics_phys = pd.DataFrame(rows_phys)

    metrics_all = pd.concat([metrics_norm, metrics_log, metrics_phys], ignore_index=True)
    metrics_all.to_csv(os.path.join(save_dir, "metrics_all_spaces.csv"), index=False)
    metrics_phys.to_csv(os.path.join(save_dir, "metrics_physical.csv"), index=False)
    metrics_log.to_csv(os.path.join(save_dir, "metrics_log_space.csv"), index=False)
    metrics_norm.to_csv(os.path.join(save_dir, "metrics_zscore.csv"), index=False)

    pred_cols = [f"pred_{x}" for x in param_names]
    true_cols = [f"true_{x}" for x in param_names]

    pred_df = pd.DataFrame(pred_phys, columns=pred_cols)
    true_df = pd.DataFrame(true_phys, columns=true_cols)

    out_pred = pd.concat([true_df, pred_df], axis=1)
    out_pred.to_csv(os.path.join(save_dir, "predictions_physical.csv"), index=False)

    print("\n=== Physical-space Evaluation Metrics ===")
    print(metrics_phys)

    print("\n=== Log-space Evaluation Metrics ===")
    print(metrics_log)

    return metrics_all


# ============================================================
# 8. 主程序
# ============================================================

def main():
    set_seed(RANDOM_SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {csv_path.resolve()}")

    # 读取和清理数据
    df = load_and_clean_dataframe(csv_path)
    print(f"清理后样本数: {len(df)}")

    if len(df) < 10:
        raise ValueError("样本数太少，无法训练。")

    # SMILES 检查
    check_smiles_tokenization(df)

    # 构建 vocab
    all_smiles = list(df["SMILES_A"]) + list(df["SMILES_B"])
    vocab = build_vocab(all_smiles)

    vocab_path = os.path.join(SAVE_DIR, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"词表已保存: {vocab_path}")
    print(f"vocab_size: {len(vocab)}")

    # 完整 dataset，用来统计 mean/std
    n_total = len(df)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(
        range(n_total),
        [n_train, n_val],
        generator=generator
    )

    train_df = df.iloc[train_subset.indices].reset_index(drop=True)
    val_df = df.iloc[val_subset.indices].reset_index(drop=True)

    cell2id = {"UNK_CELL": 0}
    for c in train_df["Cell Line"].astype(str).unique():
        if c not in cell2id:
            cell2id[c] = len(cell2id)

    with open(os.path.join(SAVE_DIR, "cell2id.json"), "w", encoding="utf-8") as f:
        json.dump(cell2id, f, ensure_ascii=False, indent=2)

    train_dataset = FlowDataset(
        df=train_df,
        vocab=vocab,
        max_len=MAX_LEN,
        train=True,
        cell2id=cell2id
    )

    val_dataset = FlowDataset(
        df=val_df,
        vocab=vocab,
        max_len=MAX_LEN,
        train=False,
        mean=train_dataset.mean,
        std=train_dataset.std,
        cell2id=cell2id
    )
    full_dataset = train_dataset

    # 保存标准化参数
    np.save(os.path.join(SAVE_DIR, "theta_mean.npy"), train_dataset.mean)
    np.save(os.path.join(SAVE_DIR, "theta_std.npy"), train_dataset.std)

    print("theta mean:", train_dataset.mean)
    print("theta std :", train_dataset.std)
    print(f"标准化参数已保存到: {SAVE_DIR}")

    # 生成 train/val indices
    n_total = len(df)
    n_val = max(1, int(n_total * VAL_RATIO))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(RANDOM_SEED)


    train_subset, val_subset_tmp = random_split(
        range(n_total),
        [n_train, n_val],
        generator=generator
    )



    # 验证集不做 A/B 交换增强，所以创建 eval_dataset
    eval_dataset = FlowDataset(
        df=df,
        vocab=vocab,
        max_len=MAX_LEN,
        train=False,
        mean=full_dataset.mean,
        std=full_dataset.std,
        cell2id=full_dataset.cell2id
    )

    val_subset = Subset(eval_dataset, val_subset_tmp.indices)



    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples  : {len(val_subset)}")
    print(f"Device       : {DEVICE}")

    # 初始化模型
    model = FlowModel(vocab_size=len(vocab),num_cells=len(full_dataset.cell2id)).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_val_loss = float("inf")
    history = []

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            for k in batch:
                batch[k] = batch[k].to(DEVICE)

            loss = flow_loss(model, batch, DEVICE)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # validation loss
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(DEVICE)

                loss = flow_loss(model, batch, DEVICE)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": scheduler.get_last_lr()[0]
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | "
            f"LR: {scheduler.get_last_lr()[0]:.8f}"
        )

        # 每个 epoch 保存
        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, f"flow_epoch{epoch}.pt")
        )

        # 保存 best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "flow_best2.pt")
            )

    # 保存训练历史
    pd.DataFrame(history).to_csv(os.path.join(SAVE_DIR, "training_history.csv"), index=False)

    print("\n训练完成。")
    print(f"Best val loss: {best_val_loss:.5f}")
    print(f"Best model saved to: {os.path.join(SAVE_DIR, 'flow_best2.pt')}")

    # 加载 best 模型做最终评估
    best_model = FlowModel(vocab_size=len(vocab),num_cells=len(full_dataset.cell2id)).to(DEVICE)
    best_model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "flow_best2.pt"), map_location=DEVICE)
    )

    evaluate_model(
        model=best_model,
        dataloader=val_loader,
        dataset_for_stats=train_dataset,
        device=DEVICE,
        save_dir=os.path.join(SAVE_DIR, "eval_best"),
        steps=EVAL_STEPS
    )




if __name__ == "__main__":
    main()

