"""Request schemas for the ADMET Predictor API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator
from rdkit import Chem

VALID_TASKS = Literal[
    "all",
    "caco2_wang",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "bbb_martini",
    "ppbr_az",
    "vdss_lombardo",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "cyp2c9_veith",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "half_life_obach",
    "clearance_microsomal_az",
    "clearance_hepatocyte_az",
    "herg",
    "ames",
    "dili",
    "ld50_zhu",
]

ALL_TASK_NAMES: set[str] = {
    "caco2_wang",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "bbb_martini",
    "ppbr_az",
    "vdss_lombardo",
    "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "cyp2c9_veith",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "half_life_obach",
    "clearance_microsomal_az",
    "clearance_hepatocyte_az",
    "herg",
    "ames",
    "dili",
    "ld50_zhu",
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    smiles: Annotated[str, StringConstraints(max_length=2000, strip_whitespace=True)]
    tasks: list[str] = ["all"]
    return_uncertainty: bool = True
    return_attribution: bool = False
    applicability_domain: bool = True

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        """Parse and return canonical SMILES, raising ValueError if invalid."""
        mol = Chem.MolFromSmiles(v)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {v!r}")
        return Chem.MolToSmiles(mol, canonical=True)

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v: list[str]) -> list[str]:
        """Validate task names."""
        for task in v:
            if task != "all" and task not in ALL_TASK_NAMES:
                raise ValueError(
                    f"Unknown task {task!r}. Valid tasks: {sorted(ALL_TASK_NAMES)}"
                )
        return v


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    smiles_list: list[Annotated[str, StringConstraints(max_length=2000, strip_whitespace=True)]]
    tasks: list[str] = ["all"]
    return_uncertainty: bool = True

    @field_validator("smiles_list")
    @classmethod
    def validate_size(cls, v: list[str]) -> list[str]:
        if len(v) > 10000:
            raise ValueError("Maximum 10,000 SMILES per batch")
        if len(v) == 0:
            raise ValueError("smiles_list must contain at least one SMILES")
        return v

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v: list[str]) -> list[str]:
        for task in v:
            if task != "all" and task not in ALL_TASK_NAMES:
                raise ValueError(f"Unknown task {task!r}")
        return v


class ExplainRequest(BaseModel):
    smiles: str
    task: str

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        mol = Chem.MolFromSmiles(v)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {v!r}")
        return Chem.MolToSmiles(mol, canonical=True)

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in ALL_TASK_NAMES:
            raise ValueError(f"Unknown task {v!r}")
        return v
