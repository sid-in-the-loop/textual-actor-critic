import os
import json
import ast
import re
import multiprocessing
import pandas as pd
from typing import List, Dict, Any
import numpy as np
import torch
from datasets import load_dataset
from agent_system.environments.base import EnvironmentManagerBase, to_numpy

def reliability_guard():
    """Disable destructive functions in the worker process."""
    import builtins
    import os
    import shutil
    import subprocess
    import sys
    
    builtins.exit = None
    builtins.quit = None
    os.system = None
    os.kill = None
    os.fork = None
    shutil.rmtree = None
    subprocess.Popen = None

def _execute_code_worker(code, tests, queue):
    """Worker function for executing code in a separate process."""
    import sys
    import io
    import traceback
    
    # Disable dangerous operations
    reliability_guard()
    
    # Redirect stdout/stderr to avoid cluttering logs
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    # Standard MBPP/Apps prelude for imports and setup to avoid trivial failures
    prelude = (
        "from string import *\n"
        "from re import *\n"
        "from datetime import *\n"
        "from collections import *\n"
        "from heapq import *\n"
        "from bisect import *\n"
        "from copy import *\n"
        "from math import *\n"
        "from random import *\n"
        "from statistics import *\n"
        "from itertools import *\n"
        "from functools import *\n"
        "from operator import *\n"
        "from io import *\n"
        "from sys import *\n"
        "from json import *\n"
        "from builtins import *\n"
        "from typing import *\n"
        "import string\n"
        "import re\n"
        "import datetime\n"
        "import collections\n"
        "import heapq\n"
        "import bisect\n"
        "import copy\n"
        "import math\n"
        "import random\n"
        "import statistics\n"
        "import itertools\n"
        "import functools\n"
        "import operator\n"
        "import io\n"
        "import sys\n"
        "import json\n"
        "sys.setrecursionlimit(6*10**5)\n"
    )

    scope = {}
    total = len(tests)
    passed = 0
    failed_detail = ""
    try:
        # Combine prelude with candidate code
        full_code = prelude + code
        compiled_code = compile(full_code, '<string>', 'exec')
        
        # Execute in a shared scope for globals and locals
        exec(compiled_code, scope)
        
        # Run tests one by one
        for test in tests:
            try:
                test = test.strip()
                if not test:
                    continue
                exec(test, scope)
                passed += 1
            except Exception as exc:
                if not failed_detail:
                    failed_detail = f"{test[:160]} :: {exc}"
        
        queue.put((passed, total, failed_detail))
    except Exception as exc:
        # Syntax errors or runtime errors in the candidate code
        queue.put((0, total, f"code_error: {exc}"))

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

        # Prefer train_files; if val_files missing, fall back to train_files for eval as well.
        file_path = config.data.train_files if is_train else (config.data.val_files or config.data.train_files)
        if file_path and os.path.exists(file_path):
            print(f"[MLMT Code Env] Loading local data from {file_path}")
            if str(file_path).endswith(".csv"):
                self.df = pd.read_csv(file_path)
            else:
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
        self.current_ids = None

    @staticmethod
    def _parse_tests(tests_raw):
        """Parse tests from CSV cell into a list of strings."""
        if tests_raw is None or (isinstance(tests_raw, float) and np.isnan(tests_raw)):
            return []
        if isinstance(tests_raw, list):
            return [str(t) for t in tests_raw]
        if isinstance(tests_raw, str):
            s = tests_raw.strip()
            if not s:
                return []
            # Try JSON first
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(t) for t in parsed]
            except Exception:
                pass
            # Try Python literal
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(t) for t in parsed]
            except Exception:
                pass
            # Fallback: single test string
            return [s]
        return [str(tests_raw)]

    @staticmethod
    def _extract_code_block(code_str: str) -> str:
        """Extract code using multiple heuristics (backticks, markers, or raw)."""
        if not isinstance(code_str, str):
            return ""
            
        # 1. Handle [DONE] marker (often follows the code)
        if "[DONE]" in code_str:
            code_str = code_str.split("[DONE]")[0]
            
        # 2. Handle [BEGIN] marker (often precedes the code)
        if "[BEGIN]" in code_str:
            code_str = code_str.split("[BEGIN]")[-1]
            
        # 3. Try to find all markdown code blocks and use the LAST one (most refined)
        pattern = r"```(?:python)?\s*([\s\S]*?)```"
        blocks = re.findall(pattern, code_str, flags=re.IGNORECASE)
        if blocks:
            return blocks[-1].strip()
            
        # 4. Fallback: if no backticks but the string contains "def ", assume it's raw code
        if "def " in code_str:
            # Try to start from the first "def "
            return "def " + code_str.split("def ", 1)[-1].strip()
            
        return code_str.strip()

    def run_code_tests(self, code_str: str, tests: List[str], timeout: float = 5.0):
        """
        Execute candidate code against a list of assert-based tests with a timeout.
        Returns (passed_all: bool, pass_rate: float, failed_detail: str)
        """
        tests = tests or []
        code = self._extract_code_block(code_str)
        
        # Use multiprocessing to handle timeouts and prevent hangs (e.g., while True)
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_execute_code_worker,
            args=(code, tests, queue)
        )
        
        process.start()
        process.join(timeout=timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            return False, 0.0, f"timeout: execution exceeded {timeout}s"
        
        if queue.empty():
            return False, 0.0, "execution_error: process exited without result"
        
        passed_count, total_count, failed_detail = queue.get()
        # If no tests are found, we cannot verify correctness.
        if total_count == 0:
            return False, 0.0, failed_detail or "no_tests_found"
            
        pass_rate = passed_count / total_count
        # passed_all is only true if every test passed AND there were no execution/syntax errors
        passed_all = (passed_count == total_count) and not failed_detail.startswith("code_error")
        return passed_all, pass_rate, failed_detail

    def reset(self):
        questions, tests, ids = [], [], []
        for _ in range(self.env_num):
            if self.use_local:
                row = self.df.iloc[self.last_idx % len(self.df)]
                # Prefer MBPP-style columns; fallback to prior format
                if 'problem' in row:
                    q = str(row['problem'])
                else:
                    prompt_data = row.get('prompt', row.get('text', ''))
                    if isinstance(prompt_data, (list, np.ndarray, pd.Series)):
                        q = prompt_data[0]['content']
                    else:
                        q = str(prompt_data)

                # Handle nested data structure (e.g., from MBPP parquet)
                raw_tests = row.get('test_list')
                if raw_tests is None and 'reward_model' in row and isinstance(row['reward_model'], dict):
                    raw_tests = row['reward_model'].get('ground_truth')
                if raw_tests is None and 'extra_info' in row and isinstance(row['extra_info'], dict):
                    raw_tests = row['extra_info'].get('test_list')
                
                if raw_tests is None:
                    raw_tests = row.get('test_cases', row.get('test', ''))
                
                t = self._parse_tests(raw_tests)
                idx = str(row.get('id', row.get('task_id', self.last_idx)))

                # Extract function name and helper objects from the first test case to help the model
                if t:
                    import re
                    first_test = t[0] if isinstance(t, list) else t
                    # 1. Extract function name: assert func_name(
                    match = re.search(r'assert\s+(\w+)\s*\(', first_test)
                    if match:
                        func_name = match.group(1)
                        q = f"{q}\nYour function should be named `{func_name}`."
                        
                        # 2. Extract potential classes/helpers: Look for CamelCase or unknown names
                        # but avoid standard built-ins or the function name itself.
                        potential_objects = re.findall(r'(\b[A-Z]\w+)\(', first_test)
                        if potential_objects:
                            # Filter out duplicates and the function name itself if it's CamelCase
                            potential_objects = [obj for obj in set(potential_objects) if obj != func_name]
                            if potential_objects:
                                obj_str = ", ".join([f"`{obj}`" for obj in potential_objects])
                                q = f"{q}\nNote: The tests may use {obj_str}. Ensure any necessary classes are defined."
            else:
                item = self.dataset[self.last_idx % len(self.dataset)]
                if self.is_train:
                    q = item['text']
                    t = self._parse_tests(item['test_list'])
                    idx = str(item['task_id'])
                else:
                    q = item['prompt']
                    t = self._parse_tests(item['test'])
                    idx = str(item['task_id'])
                
                # Extract function name and helper objects from the first test case to help the model
                if t:
                    import re
                    first_test = t[0] if isinstance(t, list) else t
                    # 1. Extract function name: assert func_name(
                    match = re.search(r'assert\s+(\w+)\s*\(', first_test)
                    if match:
                        func_name = match.group(1)
                        q = f"{q}\nYour function should be named `{func_name}`."
                        
                        # 2. Extract potential classes/helpers
                        potential_objects = re.findall(r'(\b[A-Z]\w+)\(', first_test)
                        if potential_objects:
                            # Filter out duplicates and the function name itself
                            potential_objects = [obj for obj in set(potential_objects) if obj != func_name]
                            if potential_objects:
                                obj_str = ", ".join([f"`{obj}`" for obj in potential_objects])
                                q = f"{q}\nNote: The tests may use {obj_str}. Ensure any necessary classes are defined."
                
            for _ in range(self.group_n):
                questions.append(q)
                tests.append(t)
                ids.append(idx)
            
            self.last_idx += 1
            
        self.current_tests = tests
        self.current_ids = ids
        
        # Debug: log the first modified prompt to verify hint injection
        if questions:
            print(f"DEBUG: First environment prompt hint check:\n---")
            print(questions[0])
            print(f"---\n")

        infos = [{'tests': t, 'question_id': i, 'question': q} for t, i, q in zip(tests, ids, questions)]
        return {'text': questions, 'image': None, 'anchor': questions}, infos

    def step(self, text_actions: List[str]):
        rewards = []
        infos = []
        for i, code in enumerate(text_actions):
            tests = self.current_tests[i] if self.current_tests and i < len(self.current_tests) else []
            passed_all, pass_rate, failed_detail = self.run_code_tests(code, tests)
            
            # Use pass_rate as the reward for a denser signal (0.0 to 1.0)
            reward = float(pass_rate)
            rewards.append(reward)
            
            info = {
                'won': passed_all, 
                'tests_passed': passed_all, 
                'pass_rate': pass_rate,
                'task_id': self.current_ids[i] if self.current_ids else None
            }
            if failed_detail:
                info['failed_tests'] = failed_detail
            infos.append(info)

        rewards = np.array(rewards, dtype=np.float32)
        batch_size = len(text_actions)
        dones = np.ones(batch_size, dtype=bool)
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

