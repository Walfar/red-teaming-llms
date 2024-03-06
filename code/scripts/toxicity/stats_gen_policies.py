""" This file is used to compare the different generation policies by computing toxicity scores on responses and them performing some t-tests to determine whether a generation policiy maximizes the toxicity score"""

import pandas as pd 
from scipy.stats import ttest_ind
import os
from Model import Llama2ChatUncensored7B, Llama2Chat7B, RedTeamModel, Llama2Chat13B, Llama2ChatUncensored13B
from toxicity.ToxicityEval import ToxEval
import subprocess
from ast import literal_eval
import numpy as np

# fucntion to run a full evaluation in order to compute scores from input/output for teh given model and generation policy
def compute_scores(model_name,generation_policy,gpu_id):
    redteam_model = RedTeamModel(gpu_id)
    if model_name == "llama2_7b_chat":
        model = Llama2Chat7B(gpu_id)
    elif model_name == "llama2_7b_chat_uncensored":
        model = Llama2ChatUncensored7B(gpu_id)
    elif model_name == "llama2_13b_chat":
        model = Llama2Chat13B(gpu_id)
    elif model_name == "llama2_13b_chat_uncensored":
        model = Llama2ChatUncensored13B(gpu_id)
    else:
        raise(f"No such mode as {model_name}")
    
    tox_eval = ToxEval(20,redteam_model,model,gen_policy=generation_policy)
    tox_eval.run_eval()
    # mv "results/{model_name}" "results/{model_name}_{generation_policy}"
    # subprocess.run(f"mv ../results/toxicity/{model_name} ../results/toxicity/stats_tox_scores/{model_name}_{generation_policy}", shell=True, check=True)
    # IDEA: to have exact same results, save the scores computed in a seperate file, similar to saved_challenges and either load them or save them when running evals

def retrieve_scores(model_name,generation_policy,gpu_id):
    dir_path = f"../results/toxicity/stats_tox_scores/{model_name}_{generation_policy}"
    if not os.path.exists(dir_path):
        compute_scores(model_name,generation_policy,gpu_id)
    return pd.read_csv(dir_path+"/tox_scores.csv")

def tuples_to_lists(list_of_tuples):
    first_elements = []
    second_elements = []
    tup = [literal_eval(item) for item in list_of_tuples]

    for tpl in tup:
        first_elements.append(tpl[0])
        second_elements.append(tpl[1])

    return first_elements, second_elements


def compute_mean(model_name,tox_scores,generation_policy):
    print(f"Scores for {generation_policy} pn {model_name}:")

    list_of_tuples = tox_scores["score distilbert"].tolist()
    first_elements, second_elements_distilbert = tuples_to_lists(list_of_tuples)
    mean_distilbert = np.mean(second_elements_distilbert)

    list_of_tuples = tox_scores["score roberta"].tolist()
    first_elements, second_elements_roberta = tuples_to_lists(list_of_tuples)
    mean_roberta = np.mean(second_elements_roberta)

    print(f"Mean distilBERT: {mean_distilbert}")
    print(f"Mean roBERTa: {mean_roberta}")
    return second_elements_distilbert, second_elements_roberta

def ttest(score1, score2,classifier_name):
    print(f"{classifier_name}:")
    t_stat, p_value = ttest_ind(score1, score2)
    alpha = 0.05
    if p_value < alpha:
        print("There is a statistically significant difference between the means.")
    else:
        print("There is no statistically significant difference between the means.")
    print("t-statistic:", t_stat)
    print("p-value:", p_value)

def ttest_both_classifiers(model_name,generation_policy1,generation_policy2,tox_scores_gen1_distilbert,tox_scores_gen2_distilbert,tox_scores_gen1_roberta,tox_scores_gen2_roberta):
    print(f"T-test for {generation_policy1} and {generation_policy2} on {model_name}:")
    ttest(tox_scores_gen1_distilbert,tox_scores_gen2_distilbert,"distilBERT")
    ttest(tox_scores_gen1_roberta,tox_scores_gen2_roberta,"roBERTa")


def compute_mean_and_ttest(model_name,gpu_id):
    tox_scores_simple_gen = retrieve_scores(model_name,"simple_generation",gpu_id)
    tox_scores_simple_gen_distilbert, tox_scores_simple_gen_roberta = compute_mean(model_name,tox_scores_simple_gen,"simple_generation")

    tox_scores_first_tox = retrieve_scores(model_name,"first_tox",gpu_id)
    tox_scores_first_tox_distilbert, tox_scores_first_tox_roberta = compute_mean(model_name,tox_scores_first_tox,"first_tox")

    tox_scores_best_tox = retrieve_scores(model_name,"best_tox",gpu_id)
    tox_scores_best_tox_distilbert, tox_scores_best_tox_roberta = compute_mean(model_name,tox_scores_best_tox,"best_tox")

    ttest_both_classifiers(model_name,"simple_generation","first_tox",tox_scores_simple_gen_distilbert,tox_scores_first_tox_distilbert,tox_scores_simple_gen_roberta,tox_scores_first_tox_roberta)
    ttest_both_classifiers(model_name,"simple_generation","best_tox",tox_scores_simple_gen_distilbert,tox_scores_best_tox_distilbert,tox_scores_simple_gen_roberta,tox_scores_best_tox_roberta)
    ttest_both_classifiers(model_name,"first_tox","best_tox",tox_scores_best_tox_distilbert,tox_scores_first_tox_distilbert,tox_scores_best_tox_roberta,tox_scores_first_tox_roberta)

#compute_mean_and_ttest("llama2_7b_chat",0)
#compute_mean_and_ttest("llama2_7b_chat_uncensored",0)
compute_mean_and_ttest("llama2_13b_chat_uncensored",0)