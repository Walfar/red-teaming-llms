""" This file is used to determine whether there is a correlatio between the input and output scores after a toxicity evaluations using OLS regression"""

from ast import literal_eval
import statsmodels.api as sm
import pandas as pd

"""
redteam_model = RedTeamModel(0)

target_model = Llama2ChatUncensored7B(0)
attack_llama2_uncensored = ToxEval(20,redteam_model,target_model,gen_policy="simple_generation")
attack_llama2_uncensored.run_eval()

target_model = Llama2Chat7B(1)
attack_llama2_uncensored = ToxEval(20,redteam_model,target_model,gen_policy="simple_generation")
attack_llama2_uncensored.run_eval()
"""

tox_scores_llama2_13b = pd.read_csv("llama2_13b_chat/tox_scores.csv")
tox_scores_llama2_13b_uncensored = pd.read_csv("llama2_13b_chat_uncensored/tox_scores.csv")

def tuples_to_lists(list_of_tuples):
    first_elements = []
    second_elements = []
    tup = [literal_eval(item) for item in list_of_tuples]

    for tpl in tup:
        first_elements.append(tpl[0])
        second_elements.append(tpl[1])

    return first_elements, second_elements

def convert_score_to_label(df,tox):
    df[f'{tox}_input_label'] = df[f'{tox}_input_score'].apply(lambda x: 1 if x > 0.5 else 0)
    df[f'{tox}_output_label'] = df[f'{tox}_output_score'].apply(lambda x: 1 if x > 0.5 else 0)
    return df

def sep_df(df):
    list_of_tuples = df['score distilbert'].tolist()
    first_elements, second_elements = tuples_to_lists(list_of_tuples)
    df[['distilbert_input_score']] = pd.DataFrame(first_elements, index=df.index)
    df[['distilbert_output_score']] = pd.DataFrame(second_elements, index=df.index)
    df['distilbert_input_score'] = pd.to_numeric(df['distilbert_input_score'], errors='coerce')
    df['distilbert_output_score'] = pd.to_numeric(df['distilbert_output_score'], errors='coerce')

    list_of_tuples = df['score roberta'].tolist()
    first_elements, second_elements = tuples_to_lists(list_of_tuples)
    df[['roberta_input_score']] = pd.DataFrame(first_elements, index=df.index)
    df[['roberta_output_score']] = pd.DataFrame(second_elements, index=df.index)
    df['roberta_input_score'] = pd.to_numeric(df['roberta_input_score'], errors='coerce')
    df['roberta_output_score'] = pd.to_numeric(df['roberta_output_score'], errors='coerce')

    return df

def compute_OLS(df,tox):
    print(f"OLS FOR: {tox}")
    X = sm.add_constant(df['input_score'])
    model = sm.OLS(df['output_score'], X)
    results = model.fit()
    print(results.summary())
    print("===================")

def compute_correlation(df,tox):
    correlation_coefficient = df[f'{tox}_input_score'].corr(df[f'{tox}_output_score'])

    print(f"Pearson Correlation Coefficient for {tox}: {correlation_coefficient}")

def process_df(df):
    df = sep_df(df)
    df = convert_score_to_label(df,"distilbert")
    df = convert_score_to_label(df,"roberta")
    return df

def do_all(df):
    compute_OLS(df,"distilbert")
    compute_OLS(df,"roberta")
    compute_correlation(df,"distilbert")
    compute_correlation(df,"roberta")

df = process_df(tox_scores_llama2_13b)
df.to_csv("tox_scores_llama2_13b_chat.csv")
df = process_df(tox_scores_llama2_13b_uncensored)
df.to_csv("tox_scores_llama2_13b_chat_uncensored.csv")