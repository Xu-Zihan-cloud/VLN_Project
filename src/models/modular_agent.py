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
        
    def get_prompt(self, instruction: str, scene_graph: List[str]) -> str:
        available_objects = ", ".join(scene_graph[:50]) # Limit to avoid context bloat
        prompt = f"""
You are an expert robot planner in a simulated house (AI2-THOR).
Task: {instruction}

Visible Objects: [{available_objects}]

Commands allowed:
1. GoTo(object_id) - Move to an object.
2. PickUp(object_id) - Pick up a small object.
3. PutObject(object_id) - Put held object onto/inside a receptacle.
4. OpenObject(object_id) - Open drawers/fridges.
5. CloseObject(object_id) - Close drawers/fridges.
6. ToggleObject(object_id) - Turn on/off lamps/stoves.
7. CleanObject(object_id) - Use a sink to clean an object.
8. HeatObject(object_id) - Use a microwave/stove to heat an object.

Return a JSON array of strings representing the plan. 
Be logical: you must GoTo an object before interacting with it.
Example: ["GoTo(mug_1)", "PickUp(mug_1)", "GoTo(sink_1)", "CleanObject(sink_1)", "GoTo(desk_1)", "PutObject(desk_1)"]

Plan:"""
        return prompt.strip()

    def plan(self, instruction: str, scene_graph: List[str]) -> List[str]:
        prompt = self.get_prompt(instruction, scene_graph)
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config["model_name"],
                messages=[{"role": "user", "content": prompt}],
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"]
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            plan = json.loads(content)
            logger.info(f"Generated Plan: {plan}")
            return plan
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return []

    def perceive(self, controller: Any) -> List[str]:
        # For prototype, we use AI2-THOR metadata
        visible_objects = [obj['objectId'] for obj in controller.last_event.metadata['objects'] if obj['visible']]
        return visible_objects

    def execute_subtask(self, subtask: str, controller: Any) -> bool:
        try:
            action_type = subtask.split("(")[0]
            object_id = subtask.split("(")[1].replace(")", "")
            
            logger.info(f"Executing: {action_type} on {object_id}")
            
            if action_type == "GoTo":
                # Use TeleportFull for fast but correct navigation
                event = controller.step(action="TeleportFull", objectId=object_id, forceAction=False)
                return event.metadata['lastActionSuccess']
            
            # Map LLM actions to AI2-THOR ALFRED actions
            action_map = {
                "PickUp": "PickupObject",
                "PutObject": "PutObject",
                "OpenObject": "OpenObject",
                "CloseObject": "CloseObject",
                "ToggleObject": "ToggleObjectOn",
                "CleanObject": "CleanObject",
                "HeatObject": "HeatObject"
            }
            
            if action_type in action_map:
                event = controller.step(action=action_map[action_type], objectId=object_id)
                return event.metadata['lastActionSuccess']
            
            return False
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False

    def run_episode(self, controller: Any, instruction: str) -> bool:
        scene_graph = self.perceive(controller)
        plan = self.plan(instruction, scene_graph)
        if not plan: return False
        for subtask in plan:
            success = self.execute_subtask(subtask, controller)
            if not success: break
        return controller.last_event.metadata['lastActionSuccess']
