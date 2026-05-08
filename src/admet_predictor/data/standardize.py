"""SMILES standardization utilities."""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)

# SMARTS-based neutralization patterns: (charged_smarts, neutral_smarts)
_NEUTRALIZATION_PATTERNS: list[tuple[str, str]] = [
    ("[NH2+]", "[NH2]"),
    ("[NH3+]", "[NH3]"),
    ("[NH+]", "[NH]"),
    ("[n+]", "[n]"),
    ("[OH+]", "[OH]"),
    ("[O-]", "[OH]"),
    ("[S-]", "[SH]"),
    ("[N-]", "[NH]"),
    ("[n-]", "[nH]"),
    ("[P+]", "[P]"),
    ("[s+]", "[s]"),
    ("[o+]", "[o]"),
]

# Pre-compile patterns
_COMPILED_PATTERNS: list[tuple] = []
for _charged, _neutral in _NEUTRALIZATION_PATTERNS:
    _q = Chem.MolFromSmarts(_charged)
    _r = Chem.MolFromSmiles(_neutral, sanitize=False)
    if _q is not None and _r is not None:
        _COMPILED_PATTERNS.append((_q, _r, _charged, _neutral))


def _remove_salts(mol: Chem.Mol) -> Chem.Mol:
    """Keep the largest fragment by heavy atom count."""
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def _neutralize(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize common ionic groups via SMARTS replacement."""
    for query, replacement, charged, neutral in _COMPILED_PATTERNS:
        while mol.HasSubstructMatch(query):
            rw = Chem.RWMol(mol)
            match = rw.GetSubstructMatch(query)
            if not match:
                break
            # Replace the first matching atom's formal charge
            atom_idx = match[0]
            atom = rw.GetAtomWithIdx(atom_idx)
            # Adjust formal charge based on pattern
            if "[NH2+]" == charged or "[NH3+]" == charged or "[NH+]" == charged:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)
            elif "[n+]" == charged or "[s+]" == charged or "[o+]" == charged:
                atom.SetFormalCharge(0)
            elif "[O-]" == charged:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(1)
            elif "[S-]" == charged or "[N-]" == charged:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(1)
            elif "[n-]" == charged:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(1)
            elif "[OH+]" == charged:
                atom.SetFormalCharge(0)
            elif "[P+]" == charged:
                atom.SetFormalCharge(0)
            else:
                atom.SetFormalCharge(0)
            try:
                Chem.SanitizeMol(rw)
                mol = rw.GetMol()
            except Exception:
                break
    return mol


def standardize_smiles(smiles: str) -> str | None:
    """Standardize a SMILES string.

    Steps:
    1. Parse with RDKit
    2. Remove salts (keep largest fragment)
    3. Neutralize common charges
    4. Return canonical SMILES

    Returns None if any step fails.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        # Remove salts
        mol = _remove_salts(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None

        # Neutralize
        try:
            mol = _neutralize(mol)
        except Exception:
            pass  # Proceed even if neutralization fails

        # Sanitize
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        return Chem.MolToSmiles(mol, canonical=True)

    except Exception as e:
        logger.debug("standardize_smiles failed for %r: %s", smiles, e)
        return None


def standardize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """Apply standardize_smiles to each row, dropping rows that fail.

    Parameters
    ----------
    df:
        Input DataFrame containing a SMILES column.
    smiles_col:
        Name of the SMILES column.

    Returns
    -------
    DataFrame with the SMILES column replaced by canonical SMILES,
    failed rows dropped, and index reset.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame")

    original_len = len(df)
    df = df.copy()
    df[smiles_col] = df[smiles_col].map(standardize_smiles)
    df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
    dropped = original_len - len(df)
    if dropped > 0:
        logger.info("Dropped %d/%d rows during standardization", dropped, original_len)
    return df
