import os
# --- CPU Performance Optimizations ---
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"
os.environ["OPENBLAS_NUM_THREADS"] = "64"

import hydra
import torch
import logging
import pyrootutils
from omegaconf import DictConfig
from ai2thor.controller import Controller
import cv2
from typing import List, Dict, Any

# Setup root directory
pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

# Setup logging
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig):
    # Determine agent type from config
    agent_type = "e2e" if "e2e" in cfg.model._target_ else "modular"
    logger.info(f"Evaluating {agent_type} agent on real dataset (Headless Mode)...")

    # Instantiate Agent
    agent = hydra.utils.instantiate(cfg.model)

    # Init Lightning DataModule
    logger.info("Initializing DataModule...")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test") # ONLY load validation data
    val_set = datamodule.val_seen_set
    
    logger.info(f"Loaded {len(val_set)} validation episodes. Starting simulator...")

    # Initialize AI2-THOR Controller with CloudRendering and Debugging
    logger.info("Initializing AI2-THOR Controller (this may take a few minutes on first run)...")
    try:
        controller = Controller(
            agentMode="arm", 
            visibilityDistance=1.5,
            gridSize=0.25,
            renderInstanceSegmentation=True,
            width=600,
            height=600,
            platform="CloudRendering",
            # Enable internal logs to see where it hangs
            verbose=True 
        )
    except Exception as e:
        logger.warning(f"CloudRendering failed, trying default platform: {e}")
        controller = Controller(
            agentMode="arm", 
            visibilityDistance=1.5,
            gridSize=0.25,
            renderInstanceSegmentation=True,
            width=600,
            height=600
        )

    # Metrics
    results = {"success_rate": 0.0, "episodes": 0}
    max_episodes = 20 
    
    for i in range(min(len(val_set), max_episodes)):
        sample = val_set[i]
        # Standard ALFRED scenes are like FloorPlan_Train_1 etc.
        # For now we use the first available scene or FloorPlan1
        scene_id = "FloorPlan1" 
        instruction = sample.get("raw_instr", "Complete the task.")
        
        logger.info(f"Episode {i+1}/{max_episodes} | Task: {instruction}")
        controller.reset(scene=scene_id)
        
        # Run Episode
        success = agent.run_episode(controller, instruction) if agent_type == "modular" else False

        results["episodes"] += 1
        if success:
             results["success_rate"] += 1
             logger.info(f"Result: SUCCESS")
        else:
             logger.info(f"Result: FAILED")

    final_sr = results["success_rate"] / results["episodes"]
    logger.info("==========================================")
    logger.info(f"Final Success Rate: {final_sr:.2%}")
    logger.info("==========================================")

if __name__ == "__main__":
    evaluate()
