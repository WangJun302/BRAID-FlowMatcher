# data_preprocess.py
import os
import math
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem import rdmolops
from tqdm import tqdm

# Settings you can tune
FPSIZE = 1024   # Morgan fingerprint size
RADIUS = 2

OUT_FILE = "processed_data.pt"

# atomic feature helper: one-hot for common atom types + degree + formal charge + aromatic
ATOM_LIST = ["H","C","N","O","F","P","S","Cl","Br","I"]  # 常见元素，未出现的会归为 "other"
MAX_DEGREE = 6

def atom_feature(atom):
    # returns numpy array as atomic features
    sym = atom.GetSymbol()
    onehot = [1.0 if sym==s else 0.0 for s in ATOM_LIST]
    if sym not in ATOM_LIST:
        onehot.append(1.0)  # 'other' flag
    else:
        onehot.append(0.0)
    deg = atom.GetDegree()
    deg_onehot = [1.0 if deg==i else 0.0 for i in range(MAX_DEGREE+1)]
    charge = atom.GetFormalCharge()
    aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    return np.array(onehot + deg_onehot + [charge, aromatic], dtype=np.float32)

def mol_to_graph_and_coords(mol, use_3d=True):
    """
    mol: RDKit Mol (should have Hs)
    returns:
      atom_feats: [N, F]
      edge_index: [2, E]
      edge_dist: [E] (float)
      coords: [N,3] (float) -- if 3D not available, attempts to use rdMolOps.Get3DConformer? otherwise zeros
    """
    mol = Chem.AddHs(mol, addCoords=True)
    N = mol.GetNumAtoms()
    atom_feats = []
    for a in mol.GetAtoms():
        atom_feats.append(atom_feature(a))
    atom_feats = np.stack(atom_feats, axis=0)

    # coords
    conf = None
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
    coords = np.zeros((N,3), dtype=np.float32)
    if conf is not None:
        for i in range(N):
            pos = conf.GetAtomPosition(i)
            coords[i,:] = [pos.x, pos.y, pos.z]
    else:
        # fallback: zeros
        coords = np.zeros((N,3), dtype=np.float32)

    # adjacency and edges (undirected)
    edge_index = []
    edge_dist = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i,j])
        edge_index.append([j,i])
        # distance
        d = float(np.linalg.norm(coords[i]-coords[j]))
        edge_dist.append(d)
        edge_dist.append(d)
    if len(edge_index)==0:
        edge_index = np.zeros((2,0), dtype=np.int64)
        edge_dist = np.zeros((0,), dtype=np.float32)
    else:
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_dist = np.array(edge_dist, dtype=np.float32)
    return atom_feats, edge_index, edge_dist, coords

def gen_3d(mol, max_attempts=5):
    # try to generate 3D coords for RDKit mol
    try:
        molH = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        res = AllChem.EmbedMolecule(molH, params)
        if res != 0:
            # try multiple times
            for attempt in range(1, max_attempts):
                res = AllChem.EmbedMolecule(molH, params)
                if res==0:
                    break
        if res == 0:
            AllChem.UFFOptimizeMolecule(molH)
            # remove Hs will keep coordinates on heavy atoms; but we added Hs in mol_to_graph_and_coords anyway
            return molH
        else:
            return mol  # fallback
    except Exception as e:
        return mol

def mol_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except:
        return None

def compute_fingerprint(mol, nBits=FPSIZE, radius=RADIUS):
    # Morgan fingerprint as numpy float32 array
    arr = np.zeros((nBits,), dtype=np.int8)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    DataStructs = None
    try:
        from rdkit.DataStructs import ConvertToNumpyArray
        tmp = np.zeros((nBits,), dtype=np.int8)
        ConvertToNumpyArray(fp, tmp)
        return tmp.astype(np.float32)
    except Exception as e:
        # fallback: build from bits
        bits = list(fp.GetOnBits())
        for b in bits:
            if b < nBits:
                arr[b] = 1
        return arr.astype(np.float32)

def process_row(smiles_a, smiles_b, label):
    # returns dict for one sample, with two molecule graphs + fingerprints + label
    mol_a = mol_from_smiles(smiles_a)
    mol_b = mol_from_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    # try to add 3D
    mol_a3 = gen_3d(mol_a)
    mol_b3 = gen_3d(mol_b)
    # compute graphs and coords (these functions add Hs again internally)
    a_af, a_ei, a_ed, a_coords = mol_to_graph_and_coords(mol_a3)
    b_af, b_ei, b_ed, b_coords = mol_to_graph_and_coords(mol_b3)
    # fingerprints (use original mol without Hs)
    try:
        from rdkit.DataStructs import ConvertToNumpyArray
        import rdkit.DataStructs as DataStructs
    except:
        pass
    a_fp = compute_fingerprint(mol_a)
    b_fp = compute_fingerprint(mol_b)
    sample = {
        "a_atom_feats": a_af,      # numpy
        "a_edge_index": a_ei,      # numpy [2,E]
        "a_edge_dist": a_ed,       # numpy [E]
        "a_coords": a_coords,      # numpy [N,3]
        "a_fp": a_fp,              # numpy [FPSIZE]
        "b_atom_feats": b_af,
        "b_edge_index": b_ei,
        "b_edge_dist": b_ed,
        "b_coords": b_coords,
        "b_fp": b_fp,
        "label": float(label)
    }
    return sample

def main(tsv_path="screening_data_with_smiles.tsv", out_file=OUT_FILE, max_samples=None):
    df = pd.read_csv(tsv_path, sep="\t", header=0)
    samples = []
    print("Total rows:", len(df))
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sa = str(row['DRUG_SMILES']).strip()
        sb = str(row['EXCIPIENT_SMILES']).strip()
        lab = row['CLASS']
        s = process_row(sa, sb, lab)
        if s is not None:
            samples.append(s)
        if max_samples and len(samples) >= max_samples:
            break
    print("Processed samples:", len(samples))
    torch.save(samples, out_file)
    print("Saved processed data to", out_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", default="screening_data_with_smiles.tsv")
    parser.add_argument("--out", default="out processed_data.pt")
    parser.add_argument("--max", type=int, default=1876, help="max samples (0 for all)")
    args = parser.parse_args()
    main(tsv_path=args.tsv, out_file=args.out, max_samples=(args.max or None))
