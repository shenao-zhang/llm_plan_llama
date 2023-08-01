"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
#import openai
import importlib
import alfworld
import alfworld.agents.environment
from env_history import EnvironmentHistory
from typing import List, Dict, Any, Tuple
from statistics import mean
import math
import copy
from tenacity import retry, stop_after_attempt, retry_if_exception_type, retry_if_not_exception_type
import time
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from pathlib import Path
import torch, re
import torch.distributed as dist
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from fastchat.model import load_model


sample_per_node = 20 # 20 (12 not working)
depth = 2  # depth - 1, as the first layer is not counted
scale = 0.1
replan = False

FOLDER = './prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
VALUE_PROMPT_FILE = 'alfworld_value.json'
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

with open(os.path.join(FOLDER, VALUE_PROMPT_FILE), 'r') as f:
    value_d = json.load(f)



class LLM_model:
    def __init__(self, model_path):
        if model_path.startswith('lmsys'):
            self.model, self.tokenizer = load_model(model_path=model_path, device='cuda', num_gpus=1)
        else:
            free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
            max_memory = f'{free_in_GB - 2}GB'
            n_gpus = torch.cuda.device_count()
            max_memory = {i: max_memory for i in range(n_gpus)}
            self.model = AutoModelForCausalLM.from_pretrained(model_path, return_dict=True,
                                                            max_memory=max_memory, device_map='auto')
            print('device',max_memory, self.model.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.tokenizer.eos_id = self.tokenizer.encode('\n')[0]
        self.tokenizer.model_max_length = 2048
        self.tokenizer.truncation_side = 'left'

    @torch.no_grad()
    def get_ll(
        self,
        prefix: str,
        prompts: List[str],
    ) -> List[str]:
        bsz = len(prompts)
        prefix_tokens = self.tokenizer(prefix, return_tensors="pt")
        prompts_tokens = [self.tokenizer(x, return_tensors="pt") for x in prompts]
        max_prompt_size = max([len(t.input_ids[0]) for t in prompts_tokens])
        total_len = max_prompt_size
        tokens = torch.full((bsz, total_len), self.tokenizer.eos_id).cuda().long()

        logits = []
        for k, t in enumerate(prompts_tokens):
            tokens[k, : len(t.input_ids[0])] = torch.tensor(t.input_ids)[:self.tokenizer.model_max_length].long()
            logits.append(self.model(tokens[k:k+1, :].to(self.model.device)).logits)

    #   logits = self.model(tokens.to(self.model.device)).logits
        logits = torch.cat(logits, dim=0)
        acc_probs = torch.zeros(bsz).to(self.model.device)
        len_count = torch.zeros(bsz).to(self.model.device)
        for i in range(len(prefix_tokens.input_ids[0]), max_prompt_size):
            probs = torch.softmax(logits[:, i - 1, :], dim=-1)
            for j in range(bsz):    
                if tokens[j, i] != self.tokenizer.eos_id:
                    len_count[j] += 1
                    acc_probs[j] += torch.log(probs[j, tokens[j, i]])

        return torch.exp(acc_probs / len_count).cpu().numpy()

# llm = LLM_model(model_path="/mnt/bd/bloom-model/bloom_models/bloom-3b/") # wrong at env 2
#llm = LLM_model(model_path="/mnt/bd/bloom-model/meta_model/galactica-6.7b/")  # wrong at env 2
#llm = LLM_model(model_path="/mnt/bd/bloom-model/llama_model/llama_13B/") # half is wrong
#llm = LLM_model(model_path='/mnt/bd/bloom-model/bloom_models/bloom-3b/') # wrong at env 2
#llm = LLM_model(model_path='lmsys/vicuna-13b-v1.3') # wrong at env 2
#llm = LLM_model(model_path='/mnt/bd/bloom-model/bloom_models/bloomz-1b7')
llm = LLM_model(model_path='/mnt/bd/bloom-model/bloom_models/bloomz-560m')
#llm = LLM_model(model_path='lmsys/vicuna-7b-v1.3')  # 20

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


def value_estimation(task_class, task_name, receptacle_list, history):
    if history == []:
        return 0
    no_receptacle = True
    step_idx = 1
    while no_receptacle:
        receptacle = [loc for loc in receptacle_list if
                      (loc in history[-step_idx]['value'])]
        step_idx += 1
        no_receptacle = not receptacle
    receptacle = receptacle[-1]
    objects = re.findall(r'(\b\w+\b)\s+(\d+)', history[-1]['value'].replace(receptacle, ''))
    object_list = [' '.join(object) for object in objects]
    # temp_admaction[parent_effective_start_idx].append(f'open {receptacle}')

    hold_history = [obj['value'].replace('You pick up the', '').split('from')[0].strip()
                    for obj in history if obj['value'].startswith('You pick up the')]
    put_history = [obj['value'].replace('You put the', '').split('in/on')[0].strip()
                   for obj in history if obj['value'].startswith('You put the')]
    hold_object = hold_history[-1] if len(hold_history) > len(put_history) else None

    target_obj, target_receptacle = task_name.split('-')[1].lower(), task_name.split('-')[3].lower()
    hold_complete = hold_object is not None and (target_obj in hold_object)
    put_target = [obj['value'].startswith(f'You put the {target_obj}') and
                  obj['value'].split('the')[2].strip().startswith(f'{target_receptacle}') for obj in history]
                 # obj['value'].split('in/on the ')[1].startswith(f'{target_receptacle}') for obj in history]
    put_complete = any(put_target)
    if task_class == 'put':
        value = 1 if put_complete else 1 / 2 if hold_complete else 0
    elif task_class == 'clean':
        operate_complete = any(obj['value'].startswith(f'You clean the {target_obj}') for obj in history)
        value = 1 if (operate_complete and put_complete) else 2 / 3 if operate_complete else 1 / 3 if hold_complete else 0
     #   clean_history = [obj['value'].replace('You clean the', '').split('using')[0].strip()
      #            for obj in history if obj['value'].startswith('You clean the')]
     #   operate_object = (target_obj in clean_history)
    elif task_class == 'heat':
        operate_counteract = any([obj['value'].startswith(f'You cool the {target_obj}') for obj in history])
        operate_complete = any([obj['value'].startswith(f'You heat the {target_obj}') for obj in history])
        value = 1 if (operate_complete and put_complete and not operate_counteract) else 2 / 3 if (operate_complete and not operate_counteract) else 1 / 3 if hold_complete else 0
    elif task_class == 'cool':
        operate_counteract = any([obj['value'].startswith(f'You heat the {target_obj}') for obj in history])
        operate_complete = any([obj['value'].startswith(f'You cool the {target_obj}') for obj in history])
        value = 1 if (operate_complete and put_complete and not operate_counteract) else 2 / 3 if (operate_complete and not operate_counteract) else 1 / 3 if hold_complete else 0
    elif task_class == 'examine':
        operate_complete = any([obj['value'].startswith('You turn on the desklamp') for obj in history])
        value = (hold_complete + (hold_complete and operate_complete)) / 2
    elif task_class == 'puttwo':
        put_count = sum(put_target)
        value = 1 / 4 if (put_count == 0 and hold_complete) else 2 / 4 if (put_count == 1 and not hold_complete) \
            else 3 / 4 if (put_count == 1 and hold_complete) else 1 if (put_count == 2) else 0
    return value


def alfworld_run(env, base_prompt, memory: List[str], to_print=True, ob='', temp_envs=None, temp_envs_before_init=None, init_admaction=None, task=None, num_reset=None) -> Tuple[EnvironmentHistory, bool]:
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    receptacle_list = [init_a.replace('go to ', '') for init_a in init_admaction]
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    env_value_estimate = 0.0
    gamma = 0.9
    task_name, task_class = task
    while cur_step < 15:
#        if num_reset<2:
#            break
        temp_history = [copy.deepcopy(env_history) for _ in range(sample_per_node ** depth)]
        temp_admissible = [init_admaction for _ in range(sample_per_node ** depth)]
        temp_reward = [0.0 for _ in range(sample_per_node ** depth)]
        value_estimate = [env_value_estimate for _ in range(sample_per_node ** depth)]
        for dep in range(depth):
            layer_samples = sample_per_node ** dep
            for parent_idx in range(layer_samples):
                parent_effective_start_idx = sample_per_node ** (depth - dep) * parent_idx

                all_prompts = [str(temp_history[parent_effective_start_idx]) + "\n>" + tt for tt in temp_admissible[parent_effective_start_idx]]
                response_list = []
                for action_idx, action_prompt in enumerate(all_prompts):
                    action_prob = llm.get_ll(prefix=str(temp_history[parent_effective_start_idx]) + "\n>",
                                                    prompts=[action_prompt])
                    response_list.append((temp_admissible[parent_effective_start_idx][action_idx], action_prob[0]))
                response_list = sorted(response_list, key=lambda x: x[1], reverse=True)
              #  response_list = sorted(response_list, key=lambda x: x[1] + np.random.normal(0, 0.1, 1), reverse=True)
#                print("parent_idx", parent_idx, response_list)
                if cur_step != 0 or dep != 0:
                    value = value_estimation(task_class, task_name, receptacle_list, temp_history[parent_effective_start_idx]._history)
                else:
                    value = env_value_estimate
#                value_response = generator.generate(prompts=[str(temp_value_history[parent_effective_start_idx]) + "\n>"],
#                                                     max_gen_len=100, temperature=0.0, top_p=0.9)
#                value_response = value_response.split('>')[0].split('\n')[0].strip()
                print(response_list)
                traverse_num = min(sample_per_node, len(response_list))
                for i, (resp, prob) in enumerate(response_list[:traverse_num]):
                    effect_start_idx = parent_effective_start_idx + sample_per_node ** (depth - dep - 1) * i
                    effect_end_idx = parent_effective_start_idx + sample_per_node ** (depth - dep - 1) * (i + 1)
                    for env_id in range(effect_start_idx, effect_end_idx):
                        value_estimate[env_id] = value * gamma ** dep
                        observation, _, _, temp_info = temp_envs[env_id].step([resp])
                        observation = process_ob(observation[0])
#                        print("plantraj:", resp, observation, value)
                        admactions = temp_info['admissible_commands'][0]
                        admactions.remove('inventory') if 'inventory' in admactions else None 
                        admactions.remove('look') if 'look' in admactions else None
                        admactions = [s for s in admactions if not s.startswith('examine') ]
                        temp_admissible[env_id] = admactions
                        temp_reward[env_id] += prob * scale
                        temp_history[env_id].add("action", resp)
                        temp_history[env_id].add("observation", observation)
                        if dep == depth - 1:  # terminal value
                            value_estimate[env_id] = value_estimation(task_class, task_name, receptacle_list,
                                                                      temp_history[env_id]._history) * gamma ** dep
   #     rew_value = temp_reward + value_estimate
        rew_value = [sum(x) for x in zip(temp_reward, value_estimate)]
        argmax = rew_value.index(max(rew_value))
        if temp_reward[argmax] > value_estimate[argmax]:
            print("Cumulative reward dominates!")
        else:
            print("Value estimation dominates!")
        print(value_estimate)
        env_value_estimate = value_estimate[argmax]
        rollout = 1 if replan else (len(temp_history[argmax]._history)-len(env_history._history)) // 2
        for _ in range(rollout):
            if len(temp_history[argmax]._history) > len(env_history._history):
                action = temp_history[argmax]._history[len(env_history._history)]['value']
            else:
                action = 'skip'
            env_history.add("action", action)
            observation, reward, done, info = env.step([action])
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]

            env_history.add("observation", observation)
            if to_print:
                print(f'{cur_step}> {action}\n{observation}')
                sys.stdout.flush()
            if reward:
                return env_history, True
#            elif env_history.check_is_exhausted():
#                return env_history, False
            cur_step += 1
        init_admaction = info['admissible_commands'][0]
        for ii, tem_e in enumerate(temp_envs):
            tem_e = temp_envs_before_init[ii].init_env(batch_size=1)
            for _ in range(num_reset + 1):  # the first num_reset makes tem_e at the same environment as env
                _, _ = tem_e.reset()
            for prev_step in env_history._history:
                if prev_step['label'] == "action":
                    _, _, _, _ = tem_e.step([prev_step["value"]])
            temp_envs[ii] = tem_e
    return env_history, False

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

def run_trial(
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        use_memory: bool
    ) -> List[Dict[str, Any]]:
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)

    with open('base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"

    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    temp_envs_before_init = [copy.deepcopy(env) for _ in range(sample_per_node ** depth)]
    env = env.init_env(batch_size=1)
    temp_envs = [tem.init_env(batch_size=1) for tem in temp_envs_before_init]

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)

    for z, env_config in enumerate(env_configs):
        print(f'{z} / {len(env_configs)}')
        ob, info = env.reset()
        init_admaction = info['admissible_commands'][0][:-2]
      #  admissiable_actions = info['admissible_commands'][0]
      #  temp_admissible = [admissiable_actions for _ in range(sample_per_node ** depth)]
        for tem_e in temp_envs:
            tem_ob, tem_info = tem_e.reset()

        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

        print(f"using {name}")

        if env_config["is_success"]:
            num_successes += 1

            # log to world log
            with open(world_log_path, 'a') as wf:
                wf.write(f'Environment #{z} Trial #{trial_idx}: SUCCESS\n')
            with open(trial_log_path, 'a') as wf:
                wf.write(f'\n#####\n\nEnvironment #{z}: Success\n\n#####\n')
            continue

        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'act_{v}_1'] + d[f'act_{v}_0']
             #   value_prompt = 'You are a value critic of states in a household task. Here are two examples.\n' + value_d[f'value_{v}_1'] + value_d[f'value_{v}_0']
                final_env_history, is_success = alfworld_run(env, base_prompt, env_config["memory"] if use_memory else [],
                                                             to_print=True, ob=ob, temp_envs=temp_envs,
                                                             temp_envs_before_init=temp_envs_before_init,
                                                             init_admaction=init_admaction, task=(name, v), num_reset=z)

                # update env config
                if is_success:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                    env_configs[z]['is_success'] = True
                    num_successes += 1
                    num_additional_successes += 1
                else:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'
#                rank = dist.get_rank()

                if True:
                    # log to world log
                    with open(world_log_path, 'a') as f:
                        f.write(status_str + '\n')

                    # log env results to trial log
                    with open(trial_log_path, 'a') as wf:
                        wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

    # close environment object
    env.close()
    for tem_e in temp_envs:
        tem_e.close()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return env_configs
