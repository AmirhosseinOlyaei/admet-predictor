"""ChemBERTa-based SMILES encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class BertEncoder(nn.Module):
    """ChemBERTa encoder that maps SMILES strings to molecular embeddings.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for ChemBERTa.
    hidden_dim:
        Output dimension (projects 768-dim CLS token → hidden_dim).
    max_length:
        Maximum tokenizer length for SMILES sequences.
    """

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        hidden_dim: int = 256,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self._model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size  # typically 768

        self.proj = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def freeze(self) -> None:
        """Freeze all BERT parameters (e.g. for warm-up epochs)."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all BERT parameters for fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, smiles_list: list[str]) -> Tensor:
        """Encode a list of SMILES strings.

        Parameters
        ----------
        smiles_list:
            Batch of SMILES strings.

        Returns
        -------
        Tensor of shape [batch_size, hidden_dim].
        """
        device = next(self.parameters()).device

        encoding = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Move to same device as model
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token representation: [batch_size, bert_dim]
        cls_repr = outputs.last_hidden_state[:, 0, :]

        return self.proj(cls_repr)
