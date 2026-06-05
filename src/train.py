import os
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["VECLIB_MAXIMUM_THREADS"] = "64"
os.environ["NUMEXPR_NUM_THREADS"] = "64"

import hydra
import torch
import pyrootutils
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Setup root directory for hydra
pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # Set seed
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Init Lightning DataModule
    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    # Init Lightning Module
    print(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    # Init Loggers
    loggers = []
    if cfg.get("logger"):
        for _, lg_conf in cfg.logger.items():
            loggers.append(hydra.utils.instantiate(lg_conf))

    # Init Callbacks
    callbacks = []
    if cfg.get("callbacks"):
        for _, cb_conf in cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning Trainer
    print(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        logger=loggers,
        callbacks=callbacks
    )

    # Train the model
    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    # Test the model
    print("Starting testing...")
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    train()
