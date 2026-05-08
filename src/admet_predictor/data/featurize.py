"""RDKit-based molecular featurization for graph neural networks."""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.rdchem import (
    BondStereo,
    BondType,
    ChiralType,
    HybridizationType,
)

# ---------------------------------------------------------------------------
# Atom feature lists
# ---------------------------------------------------------------------------
ATOMIC_NUMS = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]
DEGREES = list(range(11))
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = list(range(5))
CHIRALITIES = [
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
]
HYBRIDIZATIONS = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
]

# Bond feature lists
BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
BOND_STEREOS = [
    BondStereo.STEREONONE,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOANY,
]


def _one_hot(value, choices: list) -> list[int]:
    """One-hot encode value with an extra 'other' category at the end."""
    encoding = [0] * (len(choices) + 1)
    if value in choices:
        encoding[choices.index(value)] = 1
    else:
        encoding[-1] = 1
    return encoding


def _atom_features(atom) -> list[float]:
    """Compute feature vector for a single RDKit atom."""
    features: list[int] = []
    features += _one_hot(atom.GetAtomicNum(), ATOMIC_NUMS)          # 12
    features += _one_hot(atom.GetDegree(), DEGREES)                  # 12
    features += _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)     # 6
    features += _one_hot(atom.GetTotalNumHs(), NUM_HS)               # 6
    features += _one_hot(atom.GetChiralTag(), CHIRALITIES)           # 4
    features += [int(atom.GetIsAromatic())]                          # 1
    features += [int(atom.IsInRing())]                               # 1
    features += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)    # 6
    # Total = 12+12+6+6+4+1+1+6 = 48
    return features


def _bond_features(bond) -> list[float]:
    """Compute feature vector for a single RDKit bond."""
    features: list[int] = []
    features += _one_hot(bond.GetBondType(), BOND_TYPES)      # 4
    features += [int(bond.GetIsConjugated())]                  # 1
    features += [int(bond.IsInRing())]                         # 1
    features += _one_hot(bond.GetStereo(), BOND_STEREOS)       # 4
    # Total = 4+1+1+4 = 10
    return features


# Precompute feature dimensions
_dummy_mol = Chem.MolFromSmiles("C")
ATOM_FEAT_DIM = len(_atom_features(_dummy_mol.GetAtomWithIdx(0)))
_dummy_bond = _dummy_mol.GetBondWithIdx(0) if _dummy_mol.GetNumBonds() > 0 else None
# Use ethane to get a bond
_eth_mol = Chem.MolFromSmiles("CC")
BOND_FEAT_DIM = len(_bond_features(_eth_mol.GetBondWithIdx(0)))


def mol_to_graph(smiles: str) -> dict | None:
    """Convert a SMILES string to a graph dict with node/edge features.

    Returns
    -------
    dict with keys:
        node_features: torch.FloatTensor [num_atoms, ATOM_FEAT_DIM]
        edge_index:    torch.LongTensor  [2, 2*num_bonds]
        edge_features: torch.FloatTensor [2*num_bonds, BOND_FEAT_DIM]
    or None if the SMILES is invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add hydrogens implicitly (keep explicit H from SMILES)
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None

    # Node features
    node_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    node_features = torch.tensor(node_feats, dtype=torch.float)

    # Edge index + edge features (both directions)
    edge_indices: list[list[int]] = [[], []]
    edge_feats: list[list[float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)

        edge_indices[0] += [i, j]
        edge_indices[1] += [j, i]
        edge_feats += [bf, bf]

    if edge_feats:
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_features = torch.tensor(edge_feats, dtype=torch.float)
    else:
        # Molecule with no bonds (single atom)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_features = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
    }


def mol_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Return concatenated Morgan (2048-bit) + MACCS (167-bit) fingerprint.

    Returns a float32 numpy array of shape [2215] or None for invalid SMILES.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    maccs = MACCSkeys.GenMACCSKeys(mol)

    morgan_arr = np.frombuffer(morgan.ToBitString().encode(), dtype="uint8") - ord("0")
    maccs_arr = np.frombuffer(maccs.ToBitString().encode(), dtype="uint8") - ord("0")

    return np.concatenate([morgan_arr, maccs_arr]).astype(np.float32)
