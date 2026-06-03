import json
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import torch

# Configure logging
logger = logging.getLogger(__name__)

class ModularVLNAgent:
    def __init__(
        self,
        llm: Dict[str, Any],
        perception: Dict[str, Any],
        navigation: Dict[str, Any],
        **kwargs
    ):
        self.llm_config = llm
        self.perception_config = perception
        self.nav_config = navigation
        
        # Initialize OpenAI client for Ollama
        self.client = OpenAI(
            base_url=self.llm_config["base_url"],
            api_key=self.llm_config["api_key"]
        )
        
        # Perception module (e.g., Mask R-CNN) would be initialized here
        # self.perception_model = load_mask_rcnn(self.perception_config["mask_rcnn_weights"])
        
    def get_prompt(self, instruction: str, scene_graph: List[str]) -> str:
        """
        Generates a robust prompt for the LLM to decompose instructions into subtasks.
        """
        available_objects = ", ".join(scene_graph)
        prompt = f"""
You are an embodied AI assistant in a house. Your task is to decompose a high-level instruction into a sequence of low-level subtasks.

Available Objects in the scene: [{available_objects}]

Valid Subtasks:
1. GoTo(object_id): Move to the vicinity of an object.
2. PickUp(object_id): Pick up an object.
3. PutObject(object_id): Put the held object into/onto another object.
4. OpenObject(object_id): Open a drawer, fridge, etc.
5. CloseObject(object_id): Close a drawer, fridge, etc.
6. ToggleObject(object_id): Turn on/off a lamp, stove, etc.
7. SliceObject(object_id): Slice an object with a knife.

High-Level Instruction: "{instruction}"

Output the plan as a strict JSON array of strings. Do not include any explanation.
Example Output: ["GoTo(mug_1)", "PickUp(mug_1)", "GoTo(coffee_machine_1)", "PutObject(coffee_machine_1)"]

Plan:
"""
        return prompt.strip()

    def plan(self, instruction: str, scene_graph: List[str]) -> List[str]:
        """
        Calls the local LLM to generate a sequence of subtasks.
        """
        prompt = self.get_prompt(instruction, scene_graph)
        
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"]
            )
            
            content = response.choices[0].message.content.strip()
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            
            plan = json.loads(content)
            if not isinstance(plan, list):
                raise ValueError("LLM did not return a list")
                
            logger.info(f"Generated Plan: {plan}")
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return []

    def perceive(self, controller: Any) -> List[str]:
        """
        Uses the perception module (or ground truth for now) to identify objects in the scene.
        In a real scenario, this uses Mask R-CNN.
        """
        # For prototype, we use AI2-THOR metadata to get visible objects
        visible_objects = [obj['objectId'] for obj in controller.last_event.metadata['objects'] if obj['visible']]
        return visible_objects

    def execute_subtask(self, subtask: str, controller: Any) -> bool:
        """
        Executes a single subtask using deterministic navigation or interaction.
        """
        try:
            # Parse subtask: Action(Object_ID)
            action_type = subtask.split("(")[0]
            object_id = subtask.split("(")[1].replace(")", "")
            
            logger.info(f"Executing: {action_type} on {object_id}")
            
            if action_type == "GoTo":
                # AI2-THOR Navigation API
                event = controller.step(
                    action="TeleportFull",
                    objectId=object_id,
                    forceAction=False
                )
                return event.metadata['lastActionSuccess']
            
            elif action_type == "PickUp":
                event = controller.step(action="PickupObject", objectId=object_id)
                return event.metadata['lastActionSuccess']
            
            elif action_type == "PutObject":
                event = controller.step(action="PutObject", objectId=object_id)
                return event.metadata['lastActionSuccess']
                
            # Add other actions as needed...
            
            return False
        except Exception as e:
            logger.error(f"Execution of {subtask} failed: {e}")
            return False

    def run_episode(self, controller: Any, instruction: str) -> bool:
        """
        Full orchestration loop for a single VLN episode.
        """
        # 1. Initial Perception
        scene_graph = self.perceive(controller)
        
        # 2. Planning
        plan = self.plan(instruction, scene_graph)
        
        # 3. Execution
        if not plan:
            return False
            
        for subtask in plan:
            success = self.execute_subtask(subtask, controller)
            if not success:
                logger.warning(f"Failed to execute subtask: {subtask}. Attempting replan...")
                # Optional: Implement replanning here
                break
                
        # 4. Check Final Success (Task specific)
        return controller.last_event.metadata['lastActionSuccess']
