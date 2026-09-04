import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import itertools

# ================== 导入部分（保持不变）==================
try:
    from models import PairClassifier
except Exception:
    try:
        from models import PairClassifier
    except Exception as e2:
        raise ImportError("Cannot import PairClassifier from models.py or model.py") from e2

try:
    from dataset import collate_fn
except Exception as e:
    raise ImportError("Cannot import collate_fn from dataset.py") from e

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray

FPSIZE = 1024
RADIUS = 2
ATOM_LIST = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
MAX_DEGREE = 6


# ================== RDKit 预处理函数（保持你的原始实现）==================
def atom_feature(atom):
    sym = atom.GetSymbol()
    onehot = [1.0 if sym == s else 0.0 for s in ATOM_LIST]
    if sym not in ATOM_LIST:
        onehot.append(1.0)
    else:
        onehot.append(0.0)
    deg = atom.GetDegree()
    deg_onehot = [1.0 if deg == i else 0.0 for i in range(MAX_DEGREE + 1)]
    charge = atom.GetFormalCharge()
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    return np.array(onehot + deg_onehot + [charge, aromatic], dtype=np.float32)


def mol_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol if mol is not None else None
    except:
        return None


def gen_3d(mol, max_attempts=5):
    try:
        molH = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        res = AllChem.EmbedMolecule(molH, params)
        if res != 0:
            for _ in range(1, max_attempts):
                res = AllChem.EmbedMolecule(molH, params)
                if res == 0:
                    break
        if res == 0:
            AllChem.UFFOptimizeMolecule(molH)
            return molH
        return mol
    except:
        return mol


def mol_to_graph_and_coords(mol):
    mol = Chem.AddHs(mol, addCoords=True)
    N = mol.GetNumAtoms()
    atom_feats = np.stack([atom_feature(a) for a in mol.GetAtoms()], axis=0)

    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    coords = np.zeros((N, 3), dtype=np.float32)
    if conf:
        for i in range(N):
            pos = conf.GetAtomPosition(i)
            coords[i] = [pos.x, pos.y, pos.z]

    edge_index, edge_dist = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        d = float(np.linalg.norm(coords[i] - coords[j]))
        edge_dist.extend([d, d])

    edge_index = np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64)
    edge_dist = np.array(edge_dist, dtype=np.float32) if edge_dist else np.zeros(0, dtype=np.float32)

    return atom_feats, edge_index, edge_dist, coords


def compute_fingerprint(mol, nBits=FPSIZE, radius=RADIUS):
    try:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        arr = np.zeros(nBits, dtype=np.int8)
        ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)
    except:
        return np.zeros(nBits, dtype=np.float32)


def process_pair(smiles_a, smiles_b):
    mol_a = mol_from_smiles(smiles_a)
    mol_b = mol_from_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None

    mol_a3 = gen_3d(mol_a)
    mol_b3 = gen_3d(mol_b)

    a_af, a_ei, a_ed, a_coords = mol_to_graph_and_coords(mol_a3)
    b_af, b_ei, b_ed, b_coords = mol_to_graph_and_coords(mol_b3)
    a_fp = compute_fingerprint(mol_a)
    b_fp = compute_fingerprint(mol_b)

    return {
        "a_atom_feats": a_af, "a_edge_index": a_ei, "a_edge_dist": a_ed, "a_coords": a_coords, "a_fp": a_fp,
        "b_atom_feats": b_af, "b_edge_index": b_ei, "b_edge_dist": b_ed, "b_coords": b_coords, "b_fp": b_fp,
        "label": 0.0
    }


# ================== 修复后的模型加载函数（关键修复）==================
def load_checkpoint_safe(path, map_location):
    try:
        return torch.load(path, map_location=map_location)
    except RuntimeError:
        return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_ckpt(ckpt_path, device):
    ck = load_checkpoint_safe(ckpt_path, device)

    # 提取 state_dict
    if isinstance(ck, dict) and 'state_dict' in ck:
        state_dict = ck['state_dict']
        print("Loaded checkpoint with 'state_dict' key")
    else:
        state_dict = ck if isinstance(ck, dict) else ck.state_dict()
        print("Loaded raw state_dict")

    # 自动推断 hidden_dim（从 node_proj 或 fp_encoder 或 classifier）
    hidden_dim = None
    for key in list(state_dict.keys()):
        if 'node_proj.weight' in key:
            hidden_dim = state_dict[key].shape[0]
            break
        if 'fp_encoder.0.weight' in key or 'fuse_proj.0.weight' in key:
            hidden_dim = state_dict[key].shape[0]
            break
        if 'classifier.0.weight' in key:
            hidden_dim = state_dict[key].shape[1]  # 输入维度通常是 hidden_dim
            break

    # 推断 n_layers（从 gcn_layers 最大索引）
    n_layers = 2
    for key in state_dict.keys():
        if 'gcn_layers.' in key:
            try:
                layer_idx = int(key.split('gcn_layers.')[1].split('.')[0])
                n_layers = max(n_layers, layer_idx + 1)
            except:
                pass

    hidden_dim = hidden_dim or 128
    print(f"✅ Auto-inferred from checkpoint → hidden_dim={hidden_dim}, n_layers={n_layers}")

    # 使用推断的参数创建模型
    node_feat_dim = len(ATOM_LIST) + 1 + (MAX_DEGREE + 1) + 2  # 20

    model = PairClassifier(
        node_feat_dim=node_feat_dim,
        fp_dim=1024,
        hidden_dim=hidden_dim,
        n_layers=n_layers,  # 关键：使用推断的层数
        rbf_dim=32
    )

    # 加载权重（严格匹配）
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    return model

def batchify_and_predict(model, samples_list, batch_size, device, thr):
    results = []
    for i in range(0, len(samples_list), batch_size):
        sub = samples_list[i:i + batch_size]
        batch = collate_fn(sub)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.no_grad():
            out = model(batch)
            probs = torch.sigmoid(out) if (isinstance(out, torch.Tensor) and (out.min() < 0 or out.max() > 1)) else out
            probs = probs.detach().cpu().numpy().ravel()

        for p in probs:
            results.append((float(p), int(p >= thr)))
    return results


# ================== 新版主函数：跨药材分子配对 ==================
def run_cross_herb_inference(input_path, ckpt_path, out_path="herb_molecule_pairs_predictions.csv",
                             batch_size=16, device_str="cuda", thr=0.2, max_mols_per_herb=25):
    device_str = args.device.lower().strip()
    if "cuda" in device_str and torch.cuda.is_available():
        device = torch.device(device_str)
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
    elif "cuda" in device_str:
        device = torch.device("cpu")
        print("⚠️ CUDA 请求失败，已自动切换到 CPU")
    else:
        device = torch.device("cpu")
        print("使用 CPU 模式")

    print("Loading model...")

    model = build_model_from_ckpt(ckpt_path, device)

    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"Loaded {len(df)} rows from {input_path}")

    # 按 HERB 分组，提取 min/max（药材级别）
    grouped = df.groupby('HERB')
    herb_molecules = {}

    for herb, group in grouped:
        min_val = group['min'].dropna().iloc[0] if not group['min'].dropna().empty else None
        max_val = group['max'].dropna().iloc[0] if not group['max'].dropna().empty else None

        mols = []
        for _, row in group.iterrows():
            smiles = str(row['SMILES']).strip()
            if smiles and smiles.lower() not in ['nan', 'none', '']:
                mols.append({
                    'herb': herb,
                    'smiles': smiles,
                    'OB': row.get('OB'),
                    'DL': row.get('DL'),
                    'min': min_val,
                    'max': max_val
                })

        # 限制每个药材的分子数量，防止组合爆炸
        if len(mols) > max_mols_per_herb:
            print(f"⚠️ {herb} has {len(mols)} molecules → only using first {max_mols_per_herb}")
            mols = mols[:max_mols_per_herb]

        herb_molecules[herb] = mols

    print(f"Prepared {len(herb_molecules)} herbs.")

    # 生成跨药材配对（A < B，避免重复）
    all_samples = []
    all_meta = []
    herb_list = list(herb_molecules.keys())

    print("Generating cross-herb molecule pairs (this may take time)...")
    for i in tqdm(range(len(herb_list)), desc="Herb A"):
        herb_a = herb_list[i]
        mols_a = herb_molecules[herb_a]

        for j in range(i + 1, len(herb_list)):
            herb_b = herb_list[j]
            mols_b = herb_molecules[herb_b]

            for ma in mols_a:
                for mb in mols_b:
                    sample = process_pair(ma['smiles'], mb['smiles'])
                    if sample is None:
                        continue
                    all_samples.append(sample)
                    all_meta.append({
                        'herb_a': ma['herb'], 'smiles_a': ma['smiles'],
                        'OB_a': ma['OB'], 'DL_a': ma['DL'],
                        'min_a': ma['min'], 'max_a': ma['max'],
                        'herb_b': mb['herb'], 'smiles_b': mb['smiles'],
                        'OB_b': mb['OB'], 'DL_b': mb['DL'],
                        'min_b': mb['min'], 'max_b': mb['max']
                    })

    print(f"Generated {len(all_samples)} valid cross-herb pairs.")

    if not all_samples:
        raise RuntimeError("No valid pairs generated!")

    print(f"Running inference (batch_size={batch_size})...")
    preds = batchify_and_predict(model, all_samples, batch_size, device, thr)

    # 合并结果并排序
    out_rows = []
    for meta, (prob, pred) in zip(all_meta, preds):
        row = meta.copy()
        row['prob'] = prob
        row['pred'] = pred
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    out_df = out_df.sort_values(by='prob', ascending=False)  # 高概率排前面
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"✅ Done! Saved {len(out_df)} predictions to: {out_path}")
    print(f"   Positive pairs (prob >= {thr}): {out_df['pred'].sum()}")


# ================== 运行入口 ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-herb molecule self-assembly prediction")
    parser.add_argument("--input", default="hreb_smiles-utf8.csv")
    parser.add_argument("--ckpt",
                        default=r'D:\Users\admin\Desktop\generative-model-master-main\Self-assembling molecule\checkpoints\best_model_epoch43_auc0.8935.pt')
    parser.add_argument("--out", default="herb_molecule_pairs_predictions.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--thr", type=float, default=0.2)
    parser.add_argument("--max_mols", type=int, default=25, help="Max molecules per herb to avoid too many pairs")

    args = parser.parse_args()


    run_cross_herb_inference(args.input, args.ckpt, args.out,
                             batch_size=args.batch_size,
                             device_str=args.device,
                             thr=args.thr,
                             max_mols_per_herb=args.max_mols)