import os
import pandas as pd
from typing import List, Dict, Any
import numpy as np
import torch
from datasets import load_dataset
from agent_system.environments.base import EnvironmentManagerBase, to_numpy

class MLMTMathEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, config, is_train=True):
        super().__init__(None, None, config)
        self.is_train = is_train
        self.last_idx = 0
        
        file_path = config.data.train_files if is_train else config.data.val_files
        if file_path and os.path.exists(file_path):
            print(f"[MLMT Math Env] Loading local data from {file_path}")
            self.df = pd.read_parquet(file_path)
            self.use_local = True
        else:
            # Datasets
            print(f"[MLMT Math Env] Local file not found, falling back to HuggingFace")
            if is_train:
                self.dataset_name = "qwedsacf/competition_math"
                self.dataset = load_dataset(self.dataset_name, split="train")
            else:
                self.dataset_name = "HuggingFaceH4/MATH-500"
                self.dataset = load_dataset(self.dataset_name, split="test")
            self.use_local = False
            
        self.env_num = config.data.train_batch_size if is_train else config.data.val_batch_size
        self.group_n = config.env.rollout.n if (is_train and config.env.rollout.n > 0) else 1
        
        self.current_grounds = None
        self._event_loop = None

    def _get_event_loop(self):
        import asyncio
        if self._event_loop is None:
            try:
                self._event_loop = asyncio.get_event_loop()
            except RuntimeError:
                self._event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._event_loop)
        return self._event_loop

    def reset(self):
        questions, grounds, ids = [], [], []
        for _ in range(self.env_num):
            if self.use_local:
                row = self.df.iloc[self.last_idx % len(self.df)]
                # Check format from prepare_math_csv.py
                prompt_data = row['prompt']
                if isinstance(prompt_data, (list, np.ndarray, pd.Series)):
                    q = prompt_data[0]['content']
                else:
                    q = str(prompt_data)
                
                reward_model_data = row['reward_model']
                if isinstance(reward_model_data, dict):
                    g = reward_model_data.get('ground_truth', '')
                else:
                    g = row.get('answer', row.get('solution', ''))
                
                idx = str(row.get('id', self.last_idx))
            else:
                item = self.dataset[self.last_idx % len(self.dataset)]
                # Keys: qwedsacf/competition_math (problem, solution)
                # Keys: HuggingFaceH4/MATH-500 (problem, answer)
                q = item['problem']
                g = item.get('answer', item.get('solution', ''))
                idx = str(item.get('unique_id', self.last_idx))
            
            for _ in range(self.group_n):
                questions.append(q)
                grounds.append(g)
                ids.append(idx)
            
            self.last_idx += 1
            
        self.current_grounds = grounds
        infos = [{'ground_truth': g, 'question_id': i, 'question': q} for g, i, q in zip(grounds, ids, questions)]
        return {'text': questions, 'image': None, 'anchor': questions}, infos

    async def _compute_scores_async(self, text_actions: List[str]):
        from verl.utils.reward_score.math import compute_score_async
        import asyncio
        tasks = []
        for i in range(len(text_actions)):
            tasks.append(compute_score_async(text_actions[i], self.current_grounds[i], use_semantic=True))
        return await asyncio.gather(*tasks)

    def step(self, text_actions: List[str]):
        import asyncio
        loop = self._get_event_loop()
        rewards = np.array(loop.run_until_complete(self._compute_scores_async(text_actions)))
        
        batch_size = len(text_actions)
        dones = np.ones(batch_size, dtype=bool)
        infos = [{'won': rewards[i] > 0} for i in range(batch_size)]
        return {'text': text_actions, 'image': None, 'anchor': text_actions}, rewards, dones, infos

class MLMTCodeEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, config, is_train=True):
        super().__init__(None, None, config)
        self.is_train = is_train
        self.last_idx = 0
        
        file_path = config.data.train_files if is_train else config.data.val_files
        if file_path and os.path.exists(file_path):
            print(f"[MLMT Code Env] Loading local data from {file_path}")
            self.df = pd.read_parquet(file_path)
            self.use_local = True
        else:
            print(f"[MLMT Code Env] Local file not found, falling back to HuggingFace")
            if is_train:
                # Muennighoff/mbpp (task_id, text, code, test_list)
                self.dataset_name = "Muennighoff/mbpp"
                self.dataset = load_dataset(self.dataset_name, split="train")
            else:
                # openai/openai_humaneval (task_id, prompt, canonical_solution, test, entry_point)
                self.dataset_name = "openai/openai_humaneval"
                self.dataset = load_dataset(self.dataset_name, split="test")
            self.use_local = False
            
        self.env_num = config.data.train_batch_size if is_train else config.data.val_batch_size
        self.group_n = config.env.rollout.n if (is_train and config.env.rollout.n > 0) else 1
        
        self.current_tests = None

    def reset(self):
        questions, tests, ids = [], [], []
        for _ in range(self.env_num):
            if self.use_local:
                row = self.df.iloc[self.last_idx % len(self.df)]
                # Assuming code datasets have similar structure or fallback
                prompt_data = row['prompt']
                if isinstance(prompt_data, (list, np.ndarray, pd.Series)):
                    q = prompt_data[0]['content']
                else:
                    q = str(prompt_data)
                
                t = row.get('test_list', row.get('test', ''))
                idx = str(row.get('id', row.get('task_id', self.last_idx)))
            else:
                item = self.dataset[self.last_idx % len(self.dataset)]
                if self.is_train:
                    q = item['text']
                    t = item['test_list']
                    idx = str(item['task_id'])
                else:
                    q = item['prompt']
                    t = item['test']
                    idx = str(item['task_id'])
                
            for _ in range(self.group_n):
                questions.append(q)
                tests.append(t)
                ids.append(idx)
            
            self.last_idx += 1
            
        self.current_tests = tests
        infos = [{'tests': t, 'question_id': i, 'question': q} for t, i, q in zip(tests, ids, questions)]
        return {'text': questions, 'image': None, 'anchor': questions}, infos

    def step(self, text_actions: List[str]):
        # Simplified code execution reward
        rewards = []
        for i, code in enumerate(text_actions):
            if 'def' in code:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        rewards = np.array(rewards)
        batch_size = len(text_actions)
        dones = np.ones(batch_size, dtype=bool)
        infos = [{'won': rewards[i] > 0} for i in range(batch_size)]
        return {'text': text_actions, 'image': None, 'anchor': text_actions}, rewards, dones, infos

class MLMTPhysicsEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, config, is_train=True):
        super().__init__(None, None, config)
        self.is_train = is_train
        self.last_idx = 0
        
        file_path = config.data.train_files if is_train else config.data.val_files
        if file_path and os.path.exists(file_path):
            print(f"[MLMT Physics Env] Loading local data from {file_path}")
            self.df = pd.read_parquet(file_path)
            self.use_local = True
        else:
            print(f"[MLMT Physics Env] Local file not found, falling back to HuggingFace")
            if is_train:
                # camel-ai/physics (topic, sub_topic, message_1, message_2)
                self.dataset_name = "camel-ai/physics"
                self.dataset = load_dataset(self.dataset_name, split="train")
            else:
                # hendrydong/gpqa_diamond (problem, solution, domain)
                self.dataset_name = "hendrydong/gpqa_diamond"
                self.dataset = load_dataset(self.dataset_name, split="test")
            self.use_local = False
            
        self.env_num = config.data.train_batch_size if is_train else config.data.val_batch_size
        self.group_n = config.env.rollout.n if (is_train and config.env.rollout.n > 0) else 1
        
        self.current_grounds = None

    def reset(self):
        questions, grounds, ids = [], [], []
        for _ in range(self.env_num):
            if self.use_local:
                row = self.df.iloc[self.last_idx % len(self.df)]
                prompt_data = row['prompt']
                if isinstance(prompt_data, (list, np.ndarray, pd.Series)):
                    q = prompt_data[0]['content']
                else:
                    q = str(prompt_data)
                
                g = row.get('answer', row.get('solution', row.get('message_2', '')))
                idx = str(row.get('id', self.last_idx))
            else:
                item = self.dataset[self.last_idx % len(self.dataset)]
                if self.is_train:
                    q = item['message_1']
                    g = item['message_2']
                    idx = f"{item['topic']}_{item['sub_topic']}_{self.last_idx}"
                else:
                    q = item['problem']
                    g = item['solution']
                    idx = str(self.last_idx)
                
            for _ in range(self.group_n):
                questions.append(q)
                grounds.append(g)
                ids.append(idx)
            
            self.last_idx += 1
            
        self.current_grounds = grounds
        infos = [{'ground_truth': g, 'question_id': i, 'question': q} for g, i, q in zip(grounds, ids, questions)]
        return {'text': questions, 'image': None, 'anchor': questions}, infos

    def step(self, text_actions: List[str]):
        rewards = []
        for i, ans in enumerate(text_actions):
            if len(ans) > 50:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        rewards = np.array(rewards)
        batch_size = len(text_actions)
        dones = np.ones(batch_size, dtype=bool)
        infos = [{'won': rewards[i] > 0} for i in range(batch_size)]
        return {'text': text_actions, 'image': None, 'anchor': text_actions}, rewards, dones, infos

