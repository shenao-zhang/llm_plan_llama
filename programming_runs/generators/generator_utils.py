import os
import openai
import random
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)

from typing import Union, List, Optional, Callable

# openai.api_key = os.getenv("OPENAI_API_KEY")

def generic_generate_func_impl(
    func_sig: str,
    model: str,
    strategy: str,
    prev_func_impl,
    feedback,
    self_reflection,
    num_comps,
    temperature,
    REFLEXION_CHAT_INSTRUCTION: str,
    REFLEXION_FEW_SHOT: str,
    SIMPLE_CHAT_INSTRUCTION: str,
    REFLEXION_COMPLETION_INSTRUCTION: str,
    SIMPLE_COMPLETION_INSTRUCTION: str,
    fix_body: Callable[[str], str]
) -> Union[str, List[str]]:
    if strategy != "reflexion" and strategy != "simple":
        raise ValueError(
            f"Invalid strategy: given `{strategy}` but expected one of `reflexion` or `simple`")
    if strategy == "reflexion" and (prev_func_impl is None or feedback is None or self_reflection is None):
        raise ValueError(
            f"Invalid arguments: given `strategy=reflexion` but `prev_func_impl`, `feedback`, or `self_reflection` is None")

    if model == "gpt-4" or model == "gpt-3.5-turbo":
        if strategy == "reflexion":
            message = f"{REFLEXION_FEW_SHOT}\n[previous impl]:\n{prev_func_impl}\n\n[unit test results from previous impl]:\n{feedback}\n\n[reflection on previous impl]:\n{self_reflection}\n\n[improved impl]:\n{func_sig}"
            # func_bodies is a really bad name, as it can also be just 1 string
            print('----------------------- SYSTEM MESSAGE -----------------------')
            print(REFLEXION_CHAT_INSTRUCTION)
            print('----------------------------------------------')
            print(' ----------------------- USER MESSAGE -----------------------')
            print(message, flush=True)
            print('----------------------------------------------')
            func_bodies = gpt_chat(model, REFLEXION_CHAT_INSTRUCTION,
                                   message, num_comps=num_comps, temperature=temperature)
        else:
            print('----------------------- SYSTEM MESSAGE -----------------------')
            print(SIMPLE_CHAT_INSTRUCTION)
            print('----------------------------------------------')
            print(' ----------------------- USER MESSAGE -----------------------')
            print(func_sig, flush=True)
            print('----------------------------------------------')
            func_bodies = gpt_chat(model, SIMPLE_CHAT_INSTRUCTION if strategy ==
                                   "simple" else REFLEXION_CHAT_INSTRUCTION, func_sig, num_comps=num_comps, temperature=temperature)
    else:
        if strategy == "reflexion":
            prompt = f"{REFLEXION_COMPLETION_INSTRUCTION}\n{prev_func_impl}\n\nunit tests:\n{feedback}\n\nhint:\n{self_reflection}\n\n# improved implementation\n{func_sig}"
            func_bodies = gpt_completion(
                model, prompt, num_comps=num_comps, temperature=temperature)
        else:
            prompt = f"{SIMPLE_COMPLETION_INSTRUCTION}\n{func_sig}"
            func_bodies = gpt_completion(
                model, prompt, num_comps=num_comps, temperature=temperature)

    if num_comps == 1:
        assert isinstance(func_bodies, str)
        print('--------------------- GENERATED FUNC BODY ---------------------')
        print(func_sig + fix_body(func_bodies))
        print('------------------------------------------')
        return func_sig + fix_body(func_bodies)
        
    else:
        print('--------------------- GENERATED FUNC BODY ---------------------')
        print([func_sig + fix_body(func_body) for func_body in func_bodies])
        print('------------------------------------------')
        return [func_sig + fix_body(func_body) for func_body in func_bodies]


def generic_generate_internal_tests(
        func_sig: str,
        model: str,
        committee_size: int,
        max_num_tests: int,
        TEST_GENERATION_FEW_SHOT: str,
        TEST_GENERATION_CHAT_INSTRUCTION: str,
        TEST_GENERATION_COMPLETION_INSTRUCTION: str,
        parse_tests: Callable[[str], List[str]],
        is_syntax_valid: Callable[[str], bool],
        is_react: bool = False
) -> List[str]:
    """
    Generates tests for a function using a refinement technique with the number
    of specified commmittee members.
    """
    if model == "gpt-4" or model == "gpt-3.5-turbo":
        if is_react:
            message = f'{TEST_GENERATION_FEW_SHOT}\n\n[func signature]:\n{func_sig}\n\n[think]:'
            output = gpt_chat(
                model, TEST_GENERATION_CHAT_INSTRUCTION, message, max_tokens=1024)
            print(f'React test generation output: {output}')
        else:
            message = f'{TEST_GENERATION_FEW_SHOT}\n\nfunc signature:\n{func_sig}\nunit tests:'
            output = gpt_chat(
                model, TEST_GENERATION_CHAT_INSTRUCTION, message, max_tokens=1024)
    else:
        prompt = f'{TEST_GENERATION_COMPLETION_INSTRUCTION}\n\nfunc signature:\n{func_sig}\nunit tests:'
        output = gpt_completion(model, prompt, max_tokens=1024)
    all_tests = parse_tests(output)  # type: ignore
    valid_tests = [test for test in all_tests if is_syntax_valid(test)]

    # n = 3
    # first_n = min(len(valid_tests), n)
    # valid_tests = valid_tests[:first_n]

    return sample_n_random(valid_tests, max_num_tests)


def generic_generate_self_reflection(
        func: str,
        feedback: str,
        model: str,
        SELF_REFLECTION_CHAT_INSTRUCTION: str,
        SELF_REFLECTION_COMPLETION_INSTRUCTION: str,
        SELF_REFLECTION_FEW_SHOT: Optional[str] = None
) -> str:
    if model == "gpt-4" or model == "gpt-3.5-turbo":
        if SELF_REFLECTION_FEW_SHOT is not None:
            reflection = gpt_chat(
                model,
                SELF_REFLECTION_CHAT_INSTRUCTION,
                f'{SELF_REFLECTION_FEW_SHOT}\n\n[function impl]:\n{func}\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:')
            print(f'Self reflection output: {reflection}')
        else:
            reflection = gpt_chat(
                model,
                SELF_REFLECTION_CHAT_INSTRUCTION,
                f'Function implementation:\n{func}\n\nUnit test results:\n{feedback}\n\nSelf-reflection:')
    else:
        reflection = gpt_completion(
            model, f'{SELF_REFLECTION_COMPLETION_INSTRUCTION}\n{func}\n\n{feedback}\n\nExplanation:')
    return reflection  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
        model: str,
        prompt: str,
        max_tokens: int = 1024,
        stop_strs: Optional[List[str]] = None,
        temperature: float = 0.0,
        num_comps=1,
) -> Union[List[str], str]:
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


@retry(wait=wait_random_exponential(min=1, max=180), stop=stop_after_attempt(6))
def gpt_chat(
    model: str,
    system_message: str,
    user_message: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].message.content  # type: ignore

    return [choice.message.content for choice in response.choices]  # type: ignore



def sample_n_random(items: List[str], n: int) -> List[str]:
    """Sample min(n, len(items)) random items from a list"""
    assert n >= 0
    if n >= len(items):
        return items
    return random.sample(items, n)
