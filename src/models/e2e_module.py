import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from typing import Dict, Any, List, Optional

class EpisodicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        action_vocab_size: int,
        visual_feature_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float
    ):
        super().__init__()
        
        # Multimodal Encoders
        self.instr_embedding = nn.Embedding(vocab_size, d_model)
        self.visual_projection = nn.Linear(visual_feature_dim, d_model)
        
        # Position Embeddings
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model)) # Max 1000 steps
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Heads
        self.action_head = nn.Linear(d_model, action_vocab_size)
        
    def forward(self, instr_tokens, visual_features, instr_mask=None, visual_mask=None):
        # instr_tokens: [B, L_i]
        # visual_features: [B, L_v, D_v]
        
        # 1. Encode Instruction
        instr_emb = self.instr_embedding(instr_tokens) # [B, L_i, D]
        
        # 2. Project Visual Features
        visual_proj = self.visual_projection(visual_features) # [B, L_v, D]
        
        # 3. Add Position Embeddings to visual features
        batch_size, seq_len, _ = visual_proj.shape
        visual_proj = visual_proj + self.pos_encoder[:, :seq_len, :]
        
        # 4. Concatenate Multimodal Inputs
        # Combine [Instruction, Visual Sequence]
        combined_features = torch.cat([instr_emb, visual_proj], dim=1) # [B, L_i + L_v, D]
        
        # 5. Transformer Forward
        # (For BC training, we use full sequence; for inference, we'd use causal masking)
        output = self.transformer(combined_features) # [B, L_i + L_v, D]
        
        # 6. Extract Action Predictions (corresponding to visual sequence steps)
        # We predict actions based on the visual features part of the sequence
        action_logits = self.action_head(output[:, instr_emb.size(1):, :]) # [B, L_v, Action_Vocab]
        
        return action_logits

class EndToEndVLNModule(LightningModule):
    def __init__(
        self,
        vocab_size: int,
        action_vocab_size: int,
        visual_feature_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = EpisodicTransformer(
            vocab_size=vocab_size,
            action_vocab_size=action_vocab_size,
            visual_feature_dim=visual_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_sr = MeanMetric() # Success Rate
        self.val_spl = MeanMetric() # Success weighted by Path Length

    def forward(self, batch):
        return self.model(
            instr_tokens=batch["instr_tokens"],
            visual_features=batch["visual_features"]
        )

    def training_step(self, batch, batch_idx):
        action_logits = self(batch) # [B, L_v, Action_Vocab]
        
        # Flatten for loss
        loss = self.criterion(
            action_logits.view(-1, self.hparams.action_vocab_size),
            batch["action_labels"].view(-1)
        )
        
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        action_logits = self(batch)
        loss = self.criterion(
            action_logits.view(-1, self.hparams.action_vocab_size),
            batch["action_labels"].view(-1)
        )
        
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # In a real VLN validation, we would run the environment simulator here
        # to calculate actual SR and SPL. For BC training validation, we track loss.
        # SR/SPL would be updated during full evaluation episodes.
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # Simple linear warmup scheduler
        def lr_lambda(current_step: int):
            if current_step < self.hparams.warmup_steps:
                return float(current_step) / float(max(1, self.hparams.warmup_steps))
            return 1.0
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
