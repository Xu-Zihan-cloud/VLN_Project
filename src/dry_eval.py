import os
import json
import hydra
import logging
from tqdm import tqdm
from omegaconf import DictConfig
from typing import List, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def dry_evaluate(cfg: DictConfig):
    logger.info("=== Starting Logic-Only (Dry-run) Evaluation ===")
    
    # 1. Instantiate the Modular Agent (LLM Brain)
    agent = hydra.utils.instantiate(cfg.model)

    # 2. Init DataModule
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="test")
    val_set = datamodule.val_seen_set
    
    if len(val_set) == 0:
        logger.error("No validation data found.")
        return

    results = {
        "total": 0,
        "valid_plans": 0,
        "total_subtasks": 0,
        "failed_reasoning": 0
    }

    # Limit to 50 episodes for a comprehensive report
    max_episodes = 50
    pbar = tqdm(range(min(len(val_set), max_episodes)), desc="Evaluating LLM Reasoner")

    for i in pbar:
        sample = val_set[i]
        instruction = sample.get("raw_instr", "Bring the apple to the table")
        
        # In dry run, we use the expert's plan length to mock a scene graph
        # This keeps the test clean and focused on instruction parsing
        mock_scene_graph = ["mug_1", "apple_1", "microwave_1", "fridge_1", "sink_1", "table_1", "cabinet_1"]
        
        # Let LLM Plan
        plan = agent.plan(instruction, mock_scene_graph)
        
        results["total"] += 1
        if plan and isinstance(plan, list) and len(plan) > 0:
            results["valid_plans"] += 1
            results["total_subtasks"] += len(plan)
            # Basic logic check: Does it start with GoTo?
            if "GoTo" not in plan[0]:
                results["failed_reasoning"] += 1
        else:
            results["failed_reasoning"] += 1
            
        pbar.set_postfix({"ValidRate": f"{results['valid_plans']/results['total']:.2%}"})

    # Final Summary
    logger.info("\n" + "="*40)
    logger.info("LOGIC EVALUATION REPORT")
    logger.info(f"Total Tasks Processed: {results['total']}")
    logger.info(f"Successful Plan Generation: {results['valid_plans']}")
    logger.info(f"Average Plan Complexity: {results['total_subtasks']/max(1, results['valid_plans']):.2f} subtasks")
    logger.info(f"Reasoning Alignment Rate: {((results['valid_plans']-results['failed_reasoning'])/results['total']):.2%}")
    logger.info("="*40)
    logger.info("Report generated successfully. Results saved to logs.")

if __name__ == "__main__":
    dry_evaluate()
