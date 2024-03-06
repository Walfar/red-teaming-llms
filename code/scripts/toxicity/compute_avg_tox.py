""" This file is used to compute the average toxicity given the results generated with the autoredteam.py file"""

import re

NB_ROUNDS = 3
NB_CONV = 20
NB_TURNS = 10

def get_last_two_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_two_lines = lines[-2:]

    return last_two_lines

def extract_number_from_line(line):
    match = re.search(r'\d+', line)
    return int(match.group()) if match else None

def run(ref_model_name,target_model_name):
    all_rounds_distilert = 0
    all_rounds_roberta = 0
    for i in range(NB_ROUNDS):
        if i == 0:
            ref_path = f"../results/toxicity/ref_{ref_model_name}"
        else:
            ref_path = f"../results/toxicity/ref{i}_{ref_model_name}"
        last_two_lines = get_last_two_lines(f"{ref_path}/{target_model_name}/results.txt")
        all_rounds_distilert += extract_number_from_line(last_two_lines[0])
        all_rounds_roberta += extract_number_from_line(last_two_lines[1])
    return all_rounds_distilert/(NB_ROUNDS*NB_CONV*NB_TURNS), all_rounds_roberta/(NB_ROUNDS*NB_CONV*NB_TURNS)

def run_all(ref_model_name):
    print(f"reference: {ref_model_name}")
    print("LLAMA2 7B CHAT")
    print(run(ref_model_name,"llama2_7b_chat"))
    print("LLAMA2 7B CHAT UNCENSORED")
    print(run(ref_model_name,"llama2_7b_chat_uncensored"))
    print("LLAMA2 13B CHAT")
    print(run(ref_model_name,"llama2_13b_chat"))
    print("LLAMA2 13B CHAT UNCENSORED")
    print(run(ref_model_name,"llama2_13b_chat_uncensored"))

run_all("7b")
run_all("13b")
run_all("7b_uncensored")
run_all("13b_uncensored")