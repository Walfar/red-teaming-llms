""" This file was used to collect 3 full evaluations and subsequent pre-set evaluations on each target LLM for the results in the paper"""

from toxicity.ToxicityEval import ToxEval
from Model import Llama2Chat7B, RedTeamModel, Llama2ChatUncensored7B, Llama2ChatUncensored13B, Llama2Chat13B
import subprocess
from utils import create_folders_if_not_exist
import threading

lock = threading.Lock()
redteam_model = RedTeamModel(0)

def load_model(llm_name):
    """
    Loads the specified Llama2 Chat model

    Args:
        llm_name (str): The name of the Llama2 Chat model to load

    Returns:
        Llama2Chat: The loaded Llama2 Chat model
    """
    if llm_name == "7b_uncensored":
        llm = Llama2ChatUncensored7B(0)
    elif llm_name == "7b":
        llm = Llama2Chat7B(0)
    elif llm_name == "13b":
        llm = Llama2Chat13B(0)
    elif llm_name == "13b_uncensored":
        llm = Llama2ChatUncensored13B(0)
    return llm

def run_full_eval(reference_llm_name, gen_policy):
    """
    Run a full evaluation on the reference LLM with the given generation policy

    Args:
        reference_llm_name (str): The name of the reference Llama2Chat model.
        gen_policy (str): The generation policy to use.
    """
    llm = load_model(reference_llm_name)
    initial_eval = ToxEval(20, redteam_model, llm, gen_policy=gen_policy, saved_challenges_file_name="saved_challenges_" + reference_llm_name)
    initial_eval.run_eval()

def run_preset_evals(reference_llm_name, llm_name):
    """
    Run the subsequent preset evauations given the reference model by loading the prompts (challenges) that were saved for the given reference model

    Args:
        reference_llm_name (str): The name of the reference Llama2Chat model.
        llm_name (str): The name of the Llama2Chat model to evaluate.
    """
    llm = load_model(llm_name)
    eval = ToxEval(20, redteam_model, llm, is_preset=True, saved_challenges_file_name="saved_challenges_" + reference_llm_name)
    eval.run_eval()
    
def run_all(reference_llm_name, gen_policy):
    """
    Run 3 rounds of full evaluation and preset evaluations, and move results into seperate files

    Args:
        reference_llm_name (str): The name of the reference Llama2Chat model.
        gen_policy (str): The generation policy to use.
    """
    for i in range(3):
        l = ["7b_uncensored", "7b", "13b", "13b_uncensored"]
        run_full_eval(reference_llm_name, gen_policy)
        create_folders_if_not_exist(f"../results/toxicity/ref{i}_{reference_llm_name}")
        for model_name in l:
            if model_name != reference_llm_name:
                run_preset_evals(reference_llm_name, model_name)
        with lock:
            subprocess.run(f"mv ../results/toxicity/llama2_* ../results/toxicity/ref{i}_{reference_llm_name}", shell=True, check=True)

run_all("7b", "simple_generation")
run_all("7b_uncensored", "best_tox")
run_all("13b", "best_tox")
run_all("13b_uncensored", "best_tox")
