import hydra
import torch
import logging
import os
import cv2
from omegaconf import DictConfig
from ai2thor.controller import Controller
from typing import List, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

def save_frame(event, path, step):
    """Saves the current frame from AI2-THOR."""
    frame = event.frame
    frame_path = os.path.join(path, f"step_{step:03d}.png")
    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def evaluate(cfg: DictConfig):
    # Determine agent type from config
    agent_type = "e2e" if "e2e" in cfg.model._target_ else "modular"
    logger.info(f"Evaluating {agent_type} agent...")

    # Load Model/Agent
    if agent_type == "e2e":
        # Load lightning module from checkpoint
        model = hydra.utils.instantiate(cfg.model)
        # model.load_from_checkpoint(cfg.get("ckpt_path"))
        model.eval()
    else:
        # Instantiate Modular Agent
        agent = hydra.utils.instantiate(cfg.model)

    # Initialize AI2-THOR Controller
    # In practice, ALFRED/AlfWorld uses specific scene setups
    controller = Controller(
        agentMode="arm", # ALFRED uses arm mode for interactions
        visibilityDistance=1.5,
        scene="FloorPlan1",
        gridSize=0.25,
        renderInstanceSegmentation=True
    )

    # Metrics
    results = {
        "success_rate": 0.0,
        "spl": 0.0,
        "episodes": 0
    }

    # Dummy Evaluation Loop
    # In a real scenario, loop through AlfWorld validation scenes
    test_scenes = ["FloorPlan1", "FloorPlan2"] 
    
    for scene in test_scenes:
        logger.info(f"Testing Scene: {scene}")
        controller.reset(scene=scene)
        
        # High-level instruction example
        instruction = "Put a clean mug in the microwave."
        
        # Setup visualization folder
        viz_dir = f"outputs/viz/{scene}"
        os.makedirs(viz_dir, exist_ok=True)

        # Run Episode
        if agent_type == "modular":
            # Modular Agent orchestration
            scene_graph = agent.perceive(controller)
            plan = agent.plan(instruction, scene_graph)
            
            for step, subtask in enumerate(plan):
                success = agent.execute_subtask(subtask, controller)
                save_frame(controller.last_event, viz_dir, step)
                if not success:
                    break
        else:
            # E2E Agent inference (Simplified)
            # would involve tokenizing instruction and feeding visual frames sequentially
            pass

        results["episodes"] += 1
        # Check success from metadata
        if controller.last_event.metadata['lastActionSuccess']:
             results["success_rate"] += 1

    # Final tally
    final_sr = results["success_rate"] / results["episodes"]
    logger.info(f"Final Success Rate: {final_sr:.2f}")

if __name__ == "__main__":
    evaluate()
