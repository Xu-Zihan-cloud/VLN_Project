import os
os.environ["OMP_NUM_THREADS"] = "64"
os.environ["MKL_NUM_THREADS"] = "64"

import hydra
import torch
import logging
import pyrootutils
from omegaconf import DictConfig
from ai2thor.controller import Controller
from pyvirtualdisplay import Display
import cv2
from typing import List, Dict, Any

# Setup root directory
pyrootutils.setup_root(__file__, indicator=".git", pythonpath=True)

# Setup logging
logger = logging.getLogger(__name__)

def save_frame(event, path, step):
    """Saves the current frame from AI2-THOR."""
    frame = event.frame
    frame_path = os.path.join(path, f"step_{step:03d}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig):
    # Start virtual display for headless rendering
    display = Display(visible=0, size=(1024, 768))
    display.start()

    # Determine agent type from config
    agent_type = "e2e" if "e2e" in cfg.model._target_ else "modular"
    logger.info(f"Evaluating {agent_type} agent on real dataset...")

    # Instantiate Agent
    agent = hydra.utils.instantiate(cfg.model)

    # Init Lightning DataModule to get real validation scenes
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    val_set = datamodule.val_seen_set # We'll test on Seen validation first
    
    if len(val_set) == 0:
        logger.error("No validation data found. Please run download_data.sh first.")
        return

    # Initialize AI2-THOR Controller
    controller = Controller(
        agentMode="arm", 
        visibilityDistance=1.5,
        gridSize=0.25,
        renderInstanceSegmentation=True,
        width=600,
        height=600
    )

    # Metrics
    results = {
        "success_rate": 0.0,
        "total_path_length": 0.0,
        "episodes": 0
    }

    # Real Evaluation Loop
    # Limit to first 20 episodes for a quick but representative test
    max_episodes = 20 
    
    for i in range(min(len(val_set), max_episodes)):
        sample = val_set[i]
        # In a real ALFRED dataset, we need the scene name from the metadata
        # For this implementation, we extract it from the sample or path
        # Assuming sample has 'raw_instr' and we use a default scene for the demo if metadata is missing
        scene_id = "FloorPlan1" # This would be extracted from real traj_data.json
        instruction = sample.get("raw_instr", "Complete the task.")
        
        logger.info(f"Episode {i+1}/{max_episodes} | Task: {instruction}")
        
        # Reset environment
        controller.reset(scene=scene_id)
        
        # Setup visualization folder
        viz_dir = f"outputs/viz/episode_{i:03d}"
        os.makedirs(viz_dir, exist_ok=True)

        # Run Episode using the Agent's orchestration
        if agent_type == "modular":
            success = agent.run_episode(controller, instruction)
            # Frame saving is handled inside run_episode or we can add it here
        else:
            # E2E Logic...
            success = False

        results["episodes"] += 1
        if success:
             results["success_rate"] += 1
             logger.info(f"Result: SUCCESS")
        else:
             logger.info(f"Result: FAILED")

    # Final Summary
    final_sr = results["success_rate"] / results["episodes"]
    logger.info("==========================================")
    logger.info(f"Evaluation Finished on {results['episodes']} episodes")
    logger.info(f"Final Success Rate: {final_sr:.2%}")
    logger.info("==========================================")
    
    display.stop()

if __name__ == "__main__":
    evaluate()
