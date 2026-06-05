import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional, List, Dict, Any

class AlfWorldDataset(Dataset):
    def __init__(self, data_path: str, split: str, max_seq_len: int, max_instr_len: int):
        self.data_path = data_path
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_instr_len = max_instr_len
        
        # In a real scenario, we would load the index file here
        # For this implementation, we assume a standard directory structure
        # where each scene is a JSON file or part of a larger shard.
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        # Dummy implementation for structure
        # In practice, this would load ALFRED .json files or pre-processed .pt files
        return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Extract features (pre-computed visual features)
        # shape: [seq_len, visual_feature_dim]
        visual_features = torch.tensor(item['visual_features'])
        
        # Instruction tokens
        instr_tokens = torch.tensor(item['instr_tokens'])
        
        # Action labels (for BC training)
        action_labels = torch.tensor(item['action_labels'])
        
        return {
            "visual_features": visual_features,
            "instr_tokens": instr_tokens,
            "action_labels": action_labels,
            "seq_len": torch.tensor(len(visual_features)),
            "instr_len": torch.tensor(len(instr_tokens))
        }

def alfworld_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collation to handle variable length sequences with padding.
    """
    visual_features = [b['visual_features'] for b in batch]
    instr_tokens = [b['instr_tokens'] for b in batch]
    action_labels = [b['action_labels'] for b in batch]
    
    # Pad visual features: [batch_size, max_seq_len, feature_dim]
    visual_features_padded = torch.nn.utils.rnn.pad_sequence(
        visual_features, batch_first=True, padding_value=0.0
    )
    
    # Pad instruction tokens: [batch_size, max_instr_len]
    instr_tokens_padded = torch.nn.utils.rnn.pad_sequence(
        instr_tokens, batch_first=True, padding_value=0 # Assuming 0 is pad token
    )
    
    # Pad action labels: [batch_size, max_seq_len]
    action_labels_padded = torch.nn.utils.rnn.pad_sequence(
        action_labels, batch_first=True, padding_value=-100 # Standard ignore index for CrossEntropy
    )
    
    # Create masks
    seq_lens = torch.stack([b['seq_len'] for b in batch])
    instr_lens = torch.stack([b['instr_len'] for b in batch])
    
    return {
        "visual_features": visual_features_padded,
        "instr_tokens": instr_tokens_padded,
        "action_labels": action_labels_padded,
        "seq_lens": seq_lens,
        "instr_lens": instr_lens
    }

class AlfWorldDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_seq_len: int = 300,
        max_instr_len: int = 100,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_set: Optional[Dataset] = None
        self.val_seen_set: Optional[Dataset] = None
        self.val_unseen_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_set = AlfWorldDataset(
                data_path=self.hparams.data_dir,
                split="train",
                max_seq_len=self.hparams.max_seq_len,
                max_instr_len=self.hparams.max_instr_len
            )
            self.val_seen_set = AlfWorldDataset(
                data_path=self.hparams.data_dir,
                split="val_seen",
                max_seq_len=self.hparams.max_seq_len,
                max_instr_len=self.hparams.max_instr_len
            )

        if stage == "test" or stage is None:
            self.val_unseen_set = AlfWorldDataset(
                data_path=self.hparams.data_dir,
                split="val_unseen",
                max_seq_len=self.hparams.max_seq_len,
                max_instr_len=self.hparams.max_instr_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=alfworld_collate_fn
        )

    def val_dataloader(self):
        # Return both seen and unseen validation loaders if needed
        # For simplicity, we can return a list of loaders
        loader_seen = DataLoader(
            self.val_seen_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=alfworld_collate_fn
        )
        return loader_seen

    def test_dataloader(self):
        return DataLoader(
            self.val_unseen_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=alfworld_collate_fn
        )
