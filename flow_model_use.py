#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_braid_no_cell_7d.py

适用于最新训练脚本：
    flow_training_no_cell_7d.py

特点：
1. 不使用细胞系；
2. 不使用 decode；
3. theta_t 初始噪声是 7 维；
4. 直接在 7 维标准化 BRAID 参数空间做 ODE 采样；
5. 支持多次采样取均值，并输出不确定性 std。

需要同目录下或 Python 路径中有：
    flow_training_no_cell_7d.py

需要训练输出目录：
    flow_result_7d/
        vocab.json
        theta_mean.npy
        theta_std.npy
        flow_best.pt

运行：
    python predict_braid_no_cell_7d.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 必须导入最新 7D 训练脚本里的 FlowModel
from flow_model_trans import FlowModel


# ============================================================
# 1. SMILES tokenizer：必须和训练代码保持一致
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


def encode_smiles(smiles, vocab, max_len=128, warn_unk=True):
    tokens = smiles_tokenizer(smiles, strict=True)

    unk_tokens = sorted(set(t for t in tokens if t not in vocab))
    if warn_unk and len(unk_tokens) > 0:
        print(f"警告：发现训练词表外 token，将映射为 UNK: {unk_tokens}")

    ids = [vocab.get(t, 1) for t in tokens]  # 1 = UNK
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))        # 0 = PAD

    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


# ============================================================
# 2. 参数反标准化
# ============================================================

PARAM_NAMES = ["E0", "Ef", "EC50A", "EC50B", "mA", "mB", "kappa"]


def inverse_transform_theta(theta_norm, mean, std):
    """
    theta_norm: numpy array, shape [N, 7]
    mean/std: numpy array, shape [7]

    注意：
    - 第 2、3 列是 log10(EC50 + 1e-6) 空间；
    - 这里会还原成原始 uM 浓度。
    """
    theta_phys = theta_norm * std + mean

    theta_phys[:, 2] = np.power(10.0, theta_phys[:, 2]) - 1e-6
    theta_phys[:, 3] = np.power(10.0, theta_phys[:, 3]) - 1e-6
    #theta_phys[:, 6] = np.power(10.0, theta_phys[:, 6]) - 1e-6

    theta_phys[:, 2] = np.clip(theta_phys[:, 2], 0.0, None)*1000
    theta_phys[:, 3] = np.clip(theta_phys[:, 3], 0.0, None)*1000
    theta_phys[:, 2] = np.log10(theta_phys[:, 2])
    theta_phys[:, 3] = np.log10(theta_phys[:, 3])
    return theta_phys


# ============================================================
# 3. 加载模型
# ============================================================

def load_model_and_stats(
    stats_path="flow_result",
    weight_name="flow_best.pt",
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = Path(stats_path)

    vocab_path = stats_path / "vocab.json"
    mean_path = stats_path / "theta_mean.npy"
    std_path = stats_path / "theta_std.npy"
    weight_path = stats_path / weight_name

    if not vocab_path.exists():
        raise FileNotFoundError(f"找不到 vocab 文件: {vocab_path}")
    if not mean_path.exists():
        raise FileNotFoundError(f"找不到 theta_mean 文件: {mean_path}")
    if not std_path.exists():
        raise FileNotFoundError(f"找不到 theta_std 文件: {std_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"找不到模型权重: {weight_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)



    with open("flow_result/cell2id.json", "r") as f:
            cell2id = json.load(f)
    mean = np.load(mean_path)
    std = np.load(std_path)
    model = FlowModel(vocab_size=len(vocab), num_cells=len(cell2id)).to(device)
    #model = FlowModel(vocab_size=len(vocab)).to(device)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"成功加载模型: {weight_path}")
    print(f"vocab_size: {len(vocab)}")
    print(f"device: {device}")

    return model, vocab, cell2id, mean, std, device


# ============================================================
# 4. 单次 ODE 采样
# ============================================================

@torch.no_grad()
def sample_theta_once(
    model,
    vocab,
    cell2id,
    cell_line,
    smiles_a,
    smiles_b,
    score,
    device,
    steps=100,
    max_len=128
):
    model.eval()
    cell_id = cell2id[cell_line]
    a_tensor = encode_smiles(smiles_a, vocab, max_len=max_len).to(device)
    b_tensor = encode_smiles(smiles_b, vocab, max_len=max_len).to(device)
    score_tensor = torch.tensor([score], dtype=torch.float32, device=device)
    cell_tensor = torch.tensor(
        [cell_id],
        dtype=torch.long,
        device=device
    )
    batch = {
        "a": a_tensor,
        "b": b_tensor,
        "cell": cell_tensor,
        "score": score_tensor
    }

    cond, score_feat = model(batch)

    # 最新 7D flow：初始噪声是 7 维，不是 128 维
    theta_t = torch.randn(1, 7, device=device)

    dt = 1.0 / steps

    for i in range(steps):
        t = torch.ones(1, 1, device=device) * (i / steps)
        v_pred = model.flow_forward(theta_t, t, cond, score_feat)
        theta_t = theta_t + v_pred * dt

    # theta_t 本身就是 7 维标准化 BRAID 参数
    return theta_t.cpu().numpy()


# ============================================================
# 5. 多次采样预测
# ============================================================

@torch.no_grad()
def predict_braid_parameters(
    model,
    vocab,
    mean,
    std,
    cell2id,
    cell_line,
    smiles_a,
    smiles_b,
    score,
    device,
    steps=100,
    n_samples=5,
    max_len=128,
    seed=None
):
    """
    返回：
        result_mean: 7 个 BRAID 参数的均值预测
        result_std:  7 个 BRAID 参数的采样标准差，作为不确定性参考
        samples_df:  每次采样的参数结果
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    theta_norm_list = []

    for _ in range(n_samples):
        theta_norm = sample_theta_once(
            model=model,
            vocab=vocab,
            cell2id=cell2id,
            smiles_a=smiles_a,
            smiles_b=smiles_b,
            cell_line=cell_line,
            score=score,
            device=device,
            steps=steps,
            max_len=max_len
        )
        theta_norm_list.append(theta_norm)

    theta_norm_all = np.concatenate(theta_norm_list, axis=0)  # [n_samples, 7]
    theta_phys_all = inverse_transform_theta(theta_norm_all, mean, std)

    mean_vals = theta_phys_all.mean(axis=0)
    std_vals = theta_phys_all.std(axis=0)

    result_mean = {name: float(mean_vals[i]) for i, name in enumerate(PARAM_NAMES)}
    result_std = {name: float(std_vals[i]) for i, name in enumerate(PARAM_NAMES)}

    samples_df = pd.DataFrame(theta_phys_all, columns=PARAM_NAMES)

    return result_mean, result_std, samples_df


# ============================================================
# 6. 批量预测，可选
# ============================================================

def predict_from_csv(
    input_csv,
    output_csv,
    model,
    vocab,
    mean,
    std,
    device,
    steps=100,
    n_samples=1,
    max_len=128
):
    """
    输入 CSV 至少需要列：
        SMILES_A, SMILES_B, score

    输出每一行的 7 个参数均值和采样 std。
    """
    df = pd.read_csv(input_csv)
    required = ["SMILES_A", "SMILES_B", "score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"输入文件缺少必要列: {missing}")

    rows = []

    for i, row in df.iterrows():
        print(f"预测 {i + 1}/{len(df)}")

        pred_mean, pred_std, _ = predict_braid_parameters(
            model=model,
            vocab=vocab,
            mean=mean,
            std=std,
            cell_line=row["cell_line"],
            cell2id=cell2id,
            smiles_a=row["SMILES_A"],
            smiles_b=row["SMILES_B"],
            score=float(row["score"]),
            device=device,
            steps=steps,
            n_samples=n_samples,
            max_len=max_len
        )

        out = row.to_dict()

        for k, v in pred_mean.items():
            out[k] = v
        for k, v in pred_std.items():
            out[f"{k}_std"] = v

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"批量预测完成，已保存: {output_csv}")
    return out_df


# ============================================================
# 7. 示例运行
# ============================================================

if __name__ == "__main__":
    model,vocab,cell2id,mean,std,device  = load_model_and_stats(
        stats_path="flow_result",
        weight_name="flow_best.pt"
    )

    # 输入新分子的 SMILES 和 score
    smi_a = "C[C@H]1[C@@H]([C@H]([C@H]([C@@H](O1)O[C@@H]2[C@H]([C@@H]([C@H](O[C@H]2O[C@H]3C[C@@]4([C@H](C[C@H]([C@H]5[C@]4(CC[C@@H]5[C@](C)(CCC=C(C)C)O)C)O)[C@@]6([C@@H]3C([C@H](CC6)O)(C)C)C)C)CO)O)O)O)O)O"

    smi_b = "CC1=COC2=C1C(=O)C(=O)C3=C2C=CC4=C3CCCC4(C)C"

    pred_mean, pred_std, samples_df = predict_braid_parameters(
        model=model,
        vocab=vocab,
        mean=mean,
        std=std,
        smiles_a=smi_a,
        smiles_b=smi_b,
        score=0.99,
        device=device,
        steps=100,
        n_samples=50,
        max_len=128,
        cell2id=cell2id,
        cell_line="A2058"

    )

    print("\n=== BRAID 参数预测均值 ===")
    for k, v in pred_mean.items():
        if "EC50" in k:
            unit = "uM"
        elif k in ["E0", "Ef"]:
            unit = "%"
        else:
            unit = ""
        print(f"{k:8}: {v:12.5f} {unit}")

    print("\n=== 采样不确定性 std ===")
    for k, v in pred_std.items():
        if "EC50" in k:
            unit = "uM"
        elif k in ["E0", "Ef"]:
            unit = "%"
        else:
            unit = ""
        print(f"{k + '_std':12}: {v:12.5f} {unit}")

    samples_df.to_csv("single_prediction_samples.csv", index=False, encoding="utf-8-sig")
    print("\n单分子对采样结果已保存: single_prediction_samples.csv")



# import torch
# import numpy as np
# import json
# import re
# from flow_model_trans import FlowModel
#
#
# # --- 1. 基础工具函数 (需与训练代码完全一致) ---
# def smiles_tokenizer(smiles):
#     pattern = r"(\[[^\]]+]|Br|Cl|@@?|=|#|\(|\)|\.|\+|\-|\/|\\|\d+|[A-Za-z])"
#     return re.findall(pattern, smiles)
#
#
# def encode_smiles(smiles, vocab, max_len=128):
#     tokens = smiles_tokenizer(smiles)
#     ids = [vocab.get(t, 1) for t in tokens]  # 1 是 UNK
#     ids = ids[:max_len]
#     ids += [0] * (max_len - len(ids))  # 0 是 PAD
#     return torch.tensor(ids).unsqueeze(0)
#
#
# # --- 2. 核心推理函数 ---
# @torch.no_grad()
# def predict_braid_parameters(model, vocab, cell2id, mean, std, smiles_a, smiles_b, cell_line, score, device, steps=50):
#     model.eval()
#
#     # 1. 准备输入数据
#     a_tensor = encode_smiles(smiles_a, vocab).to(device)
#     b_tensor = encode_smiles(smiles_b, vocab).to(device)
#
#     cell_idx = cell2id.get(cell_line)
#     if cell_idx is None:
#         print(f"警告: 细胞系 {cell_line} 不在训练集中，使用默认索引 0")
#         cell_idx = 0
#     cell_tensor = torch.tensor([cell_idx]).to(device)
#     score_tensor = torch.tensor([score], dtype=torch.float32).to(device)
#
#     batch = {"a": a_tensor, "b": b_tensor, "cell": cell_tensor, "score": score_tensor}
#
#     # 2. 获取条件嵌入 (这里需要接收两个返回值)
#     # 根据你的训练代码，model(batch) 返回的是 (cond, score_feat)
#     cond, score_feat = model(batch)
#
#     # 3. 欧拉法求解 ODE
#     theta_t = torch.randn(1, 128).to(device)  # 注意：如果用了 encoder，初始噪声是 128 维
#     dt = 1.0 / steps
#
#     for i in range(steps):
#         t = torch.ones(1, 1).to(device) * (i / steps)
#         # ⭐ 核心修改点：传入 score_feat
#         v_pred = model.flow_forward(theta_t, t, cond, score_feat)
#         theta_t = theta_t + v_pred * dt
#
#     # 4. 解码：从 128维 映射回 7维 (此时是标准化后的 z-score)
#     theta_norm_7 = model.decode(theta_t)
#     res = theta_norm_7.cpu().numpy()[0]
#
#     # 5. 反标准化：先整体回到训练时的数值量级
#     # 此时 res_phys[2] 和 res_phys[3] 对应的是 log10(EC50)
#     res_phys = res * std + mean
#
#     # 6. 指数还原：最后将 log 空间的值转回原始物理浓度
#     # 只有在这一步，EC50 才会从 -1~2 这种小数字变成 0.1~900 这种大数字
#     res_phys[2] = 10 ** res_phys[2] - 1e-6
#     res_phys[3] = 10 ** res_phys[3] - 1e-6
#
#     # 可选：物理边界清理
#     res_phys[2] = np.clip(res_phys[2], 0, 1000)
#     res_phys[3] = np.clip(res_phys[3], 0, 1000)
#
#     # 5. 还原 log10 的参数 (EC50A, EC50B)
#     # 1. 打印模型原始输出（应该是 -1 到 1 左右）
#     print(f"1. Model Raw (z-score): {res[2]}")
#
#     # 2. 打印反标准化后的值（应该是对数尺度，-1 到 3 左右）
#     # 确保这里的 mean[2] 是那个 -0.23 左右的数，std[2] 是 1.5 左右的数
#     val_log = res[2] * std[2] + mean[2]
#     print(f"2. Log Scale Value: {val_log}")
#
#     # 3. 打印最终物理浓度（应该是 0.1 到 900）
#     val_phys = 10 ** val_log - 1e-6
#     print(f"3. Final Physical Value: {val_phys}")
#
#     return {
#         "E0": res_phys[0], "Ef": res_phys[1],
#         "EC50A": res_phys[2], "EC50B": res_phys[3],
#         "mA": res_phys[4], "mB": res_phys[5],
#         "kappa": res_phys[6]
#     }
#
# # --- 3. 执行调用 ---
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 加载必要的文件
#     stats_path = "flow_result"
#     mean = np.load(f"{stats_path}/theta_mean.npy")
#     std = np.load(f"{stats_path}/theta_std.npy")
#
#     with open(f"{stats_path}/cell2id.json", "r") as f:
#         cell2id = json.load(f)
#
#     with open("flow_result/vocab.json", "r") as f:
#         vocab = json.load(f)
#     with open("flow_result/cell2id.json", "r") as f:
#         cell2id = json.load(f)
#
#     # 初始化模型并加载权重
#     # 注意：这里的 vocab_size 需要根据你训练时的 len(vocab) 确定
#     # 建议直接从训练代码中获取 vocab
#     model = FlowModel(vocab_size=len(vocab), num_cells=len(cell2id)).to(device)
#     model.load_state_dict(torch.load("flow_result/flow_epoch70.pt", map_location=device))
#
#     # 输入数据
#
#     smi_a = "C[C@@]1(CCCN1)C2=NC3=C(C=CC=C3N2)C(=O)N.Cl.Cl"
#     smi_b = "CC(C)(C#N)C1=CC=C(C=C1)N2C3=C4C=C(C=CC4=NC=C3N(C2=O)C)C5=CC6=CC=CC=C6N=C5"
#     # 获取预测结果
#     params = predict_braid_parameters(
#         model, vocab, cell2id, mean, std,
#         smi_a, smi_b, "A2058", 0.5, device
#     )
#
#     print("\n=== BRAID 参数预测结果 ===")
#     for k, v in params.items():
#         unit = "uM" if "EC50" in k else "%" if k in ["E0", "Ef"] else ""
#         print(f"{k:8}: {v:10.4f} {unit}")
#
# # import torch
# # import numpy as np
# # import json
# # import re
# # # 确保你的 flow_model_trans.py 包含了我刚才给你的完整 FlowModel 代码
# # from flow_model_trans import FlowModel
# #
# #
# # # --- 1. 基础工具函数 ---
# # def smiles_tokenizer(smiles):
# #     pattern = r"(\[[^\]]+]|Br|Cl|@@?|=|#|\(|\)|\.|\+|\-|\/|\\|\d+|[A-Za-z])"
# #     return re.findall(pattern, smiles)
# #
# #
# # def encode_smiles(smiles, vocab, max_len=128):
# #     tokens = smiles_tokenizer(smiles)
# #     ids = [vocab.get(t, 1) for t in tokens]
# #     ids = ids[:max_len]
# #     ids += [0] * (max_len - len(ids))
# #     return torch.tensor(ids).unsqueeze(0)
# #
# #
# # # --- 2. 核心推理函数 ---
# # @torch.no_grad()
# # def predict_braid_parameters(model, vocab, cell2id, mean, std, smiles_a, smiles_b, cell_line, score, device, steps=50):
# #     model.eval()
# #
# #     # 1. 准备输入数据
# #     a_tensor = encode_smiles(smiles_a, vocab).to(device)
# #     b_tensor = encode_smiles(smiles_b, vocab).to(device)
# #     cell_idx = cell2id.get(cell_line, 0)
# #     cell_tensor = torch.tensor([cell_idx]).to(device)
# #     score_tensor = torch.tensor([score], dtype=torch.float32).to(device)
# #
# #     batch = {"a": a_tensor, "b": b_tensor, "cell": cell_tensor, "score": score_tensor}
# #
# #     # 2. 获取条件嵌入
# #     # ⭐ 修改点 1：新模型 forward 只返回一个融合后的 cond_info
# #     cond_info = model(batch)
# #
# #     # 3. 欧拉法求解 ODE
# #     # 初始噪声维度为 128 (latent_dim)
# #     theta_t = torch.randn(1, 128).to(device)
# #     dt = 1.0 / steps
# #
# #     for i in range(steps):
# #         t_val = i / steps
# #         t = torch.ones(1, 1).to(device) * t_val
# #
# #         # ⭐ 修改点 2：调用新的接口，传入融合后的 cond_info
# #         v_pred = model.flow_forward(theta_t, t, cond_info)
# #         theta_t = theta_t + v_pred * dt
# #
# #     # 4. 解码与反标准化
# #     # 使用 decode 将 128 维映射回 7 维标准化空间
# #     theta_norm_7 = model.decode(theta_t)
# #
# #     # 还原物理尺度
# #     res = theta_norm_7.cpu().numpy()[0]
# #     res_phys = res * std + mean
# #
# #     # 5. 反向变换 (对数处理的还原)
# #     # 还原 EC50A 和 EC50B (之前训练时做了 log10(x + 1e-6))
# #     res_phys[2] = 10 ** res_phys[2] - 1e-6
# #     res_phys[3] = 10 ** res_phys[3] - 1e-6
# #
# #     # 限制物理意义上的合理边界 (可选，防止预测出负的浓度)
# #     res_phys[2] = max(0, res_phys[2])
# #     res_phys[3] = max(0, res_phys[3])
# #
# #     return {
# #         "E0": res_phys[0], "Ef": res_phys[1],
# #         "EC50A": res_phys[2], "EC50B": res_phys[3],
# #         "mA": res_phys[4], "mB": res_phys[5],
# #         "kappa": res_phys[6]
# #     }
# #
# #
# # # --- 3. 执行调用 ---
# # if __name__ == "__main__":
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #
# #     stats_path = "flow_result"
# #
# #     # 加载标准化参数和映射表
# #     mean = np.load(f"{stats_path}/theta_mean.npy")
# #     std = np.load(f"{stats_path}/theta_std.npy")*10
# #     with open(f"{stats_path}/vocab.json", "r") as f:
# #         vocab = json.load(f)
# #     with open(f"{stats_path}/cell2id.json", "r") as f:
# #         cell2id = json.load(f)
# #
# #     # 初始化模型
# #     model = FlowModel(vocab_size=len(vocab), num_cells=len(cell2id)).to(device)
# #
# #     # 加载权重
# #     model_weight_path = "flow_result/flow_epoch50.pt"
# #     try:
# #         model.load_state_dict(torch.load(model_weight_path, map_location=device))
# #         print(f"✅ 成功加载模型权重: {model_weight_path}")
# #     except Exception as e:
# #         print(f"❌ 加载失败: {e}")
# #
# #     # 测试样例
# #     smi_a = "CC1=COC2=C1C(=O)C(=O)C3=C2C=CC4=C3CCCC4(C)C"
# #     smi_b = "CC(=CCC[C@@](C)([C@H]1CC[C@@]2([C@@H]1[C@@H](C[C@H]3[C@]2(C[C@@H]([C@@H]4[C@@]3(CC[C@@H](C4(C)C)O)C)O[C@H]5[C@@H]([C@H]([C@@H]([C@H](O5)CO)O)O)O[C@H]6[C@@H]([C@H]([C@@H](CO6)O)O)O)C)O)C)O[C@H]7[C@@H]([C@H]([C@@H]([C@H](O7)CO)O)O)O)C"
# #
# #     # 执行预测
# #     params = predict_braid_parameters(
# #         model, vocab, cell2id, mean, std,
# #         smi_a, smi_b, "A2058", 0.5, device, steps=100  # 增加步数可提高精度
# #     )
# #
# #     print("\n=== BRAID 参数预测结果 ===")
# #     for k, v in params.items():
# #         # 格式化输出
# #         if "EC50" in k:
# #             unit = "uM"
# #         elif k in ["E0", "Ef"]:
# #             unit = "%"
# #         else:
# #             unit = ""
# #         print(f"{k:8}: {v:10.4f} {unit}")