""" This file is used to determine if there is a significant difference between two MMLU results from two distinct LLMs using a paired t-test"""

from scipy import stats
import pandas as pd

# Data for uncensored version
data_uncensored = {
    "Task": ["hendrycksTest-abstract_algebra", "hendrycksTest-anatomy", "hendrycksTest-astronomy", 
             "hendrycksTest-business_ethics", "hendrycksTest-clinical_knowledge", "hendrycksTest-college_biology",
             "hendrycksTest-college_chemistry", "hendrycksTest-college_computer_science", 
             "hendrycksTest-college_mathematics", "hendrycksTest-college_medicine", "hendrycksTest-college_physics",
             "hendrycksTest-computer_security", "hendrycksTest-conceptual_physics", "hendrycksTest-econometrics",
             "hendrycksTest-electrical_engineering", "hendrycksTest-elementary_mathematics", "hendrycksTest-formal_logic",
             "hendrycksTest-global_facts", "hendrycksTest-high_school_biology", "hendrycksTest-high_school_chemistry",
             "hendrycksTest-high_school_computer_science", "hendrycksTest-high_school_european_history",
             "hendrycksTest-high_school_geography", "hendrycksTest-high_school_government_and_politics",
             "hendrycksTest-high_school_macroeconomics", "hendrycksTest-high_school_mathematics",
             "hendrycksTest-high_school_microeconomics", "hendrycksTest-high_school_physics",
             "hendrycksTest-high_school_psychology", "hendrycksTest-high_school_statistics",
             "hendrycksTest-high_school_us_history", "hendrycksTest-high_school_world_history", "hendrycksTest-human_aging",
             "hendrycksTest-human_sexuality", "hendrycksTest-international_law", "hendrycksTest-jurisprudence",
             "hendrycksTest-logical_fallacies", "hendrycksTest-machine_learning", "hendrycksTest-management",
             "hendrycksTest-marketing", "hendrycksTest-medical_genetics", "hendrycksTest-miscellaneous",
             "hendrycksTest-moral_disputes", "hendrycksTest-moral_scenarios", "hendrycksTest-nutrition",
             "hendrycksTest-philosophy", "hendrycksTest-prehistory", "hendrycksTest-professional_accounting",
             "hendrycksTest-professional_law", "hendrycksTest-professional_medicine", "hendrycksTest-professional_psychology",
             "hendrycksTest-public_relations", "hendrycksTest-security_studies", "hendrycksTest-sociology",
             "hendrycksTest-us_foreign_policy", "hendrycksTest-virology", "hendrycksTest-world_religions"],
    "Value": [0.3400, 0.5111, 0.5263, 0.4800, 0.5811, 0.5486, 0.3800, 0.3700, 0.3800, 0.4509, 0.2647, 0.6600,
              0.4128, 0.2544, 0.4897, 0.3042, 0.2778, 0.3600, 0.6484, 0.4532, 0.5300, 0.6424, 0.6263, 0.7513, 0.4718,
              0.2704, 0.5252, 0.2980, 0.6936, 0.3704, 0.7402, 0.7257, 0.6054, 0.6870, 0.7190, 0.6574, 0.6933, 0.3036,
              0.6699, 0.7650, 0.5100, 0.7369, 0.5838, 0.2469, 0.6111, 0.6463, 0.6389, 0.4007, 0.4224, 0.5147, 0.5359,
              0.6000, 0.6082, 0.7612, 0.8200, 0.4518, 0.7135]
}

# Data for safety-tuned version
data_safety_tuned = {
    "Task": ["hendrycksTest-abstract_algebra", "hendrycksTest-anatomy", "hendrycksTest-astronomy", 
             "hendrycksTest-business_ethics", "hendrycksTest-clinical_knowledge", "hendrycksTest-college_biology",
             "hendrycksTest-college_chemistry", "hendrycksTest-college_computer_science", 
             "hendrycksTest-college_mathematics", "hendrycksTest-college_medicine", "hendrycksTest-college_physics",
             "hendrycksTest-computer_security", "hendrycksTest-conceptual_physics", "hendrycksTest-econometrics",
             "hendrycksTest-electrical_engineering", "hendrycksTest-elementary_mathematics", "hendrycksTest-formal_logic",
             "hendrycksTest-global_facts", "hendrycksTest-high_school_biology", "hendrycksTest-high_school_chemistry",
             "hendrycksTest-high_school_computer_science", "hendrycksTest-high_school_european_history",
             "hendrycksTest-high_school_geography", "hendrycksTest-high_school_government_and_politics",
             "hendrycksTest-high_school_macroeconomics", "hendrycksTest-high_school_mathematics",
             "hendrycksTest-high_school_microeconomics", "hendrycksTest-high_school_physics",
             "hendrycksTest-high_school_psychology", "hendrycksTest-high_school_statistics",
             "hendrycksTest-high_school_us_history", "hendrycksTest-high_school_world_history", "hendrycksTest-human_aging",
             "hendrycksTest-human_sexuality", "hendrycksTest-international_law", "hendrycksTest-jurisprudence",
             "hendrycksTest-logical_fallacies", "hendrycksTest-machine_learning", "hendrycksTest-management",
             "hendrycksTest-marketing", "hendrycksTest-medical_genetics", "hendrycksTest-miscellaneous",
             "hendrycksTest-moral_disputes", "hendrycksTest-moral_scenarios", "hendrycksTest-nutrition",
             "hendrycksTest-philosophy", "hendrycksTest-prehistory", "hendrycksTest-professional_accounting",
             "hendrycksTest-professional_law", "hendrycksTest-professional_medicine", "hendrycksTest-professional_psychology",
             "hendrycksTest-public_relations", "hendrycksTest-security_studies", "hendrycksTest-sociology",
             "hendrycksTest-us_foreign_policy", "hendrycksTest-virology", "hendrycksTest-world_religions"],
    "Value": [0.3000, 0.5185, 0.5435, 0.3704, 0.6170, 0.5290, 0.3333, 0.3333, 0.4194, 0.4615, 0.2353, 0.6700,
              0.4365, 0.2683, 0.5128, 0.2727, 0.2500, 0.3750, 0.6709, 0.4593, 0.5100, 0.6837, 0.6364, 0.7692, 0.4301,
              0.2969, 0.5690, 0.3214, 0.7196, 0.3455, 0.7564, 0.7317, 0.6170, 0.6944, 0.7135, 0.6897, 0.7257, 0.3077,
              0.6944, 0.7692, 0.5128, 0.7500, 0.5970, 0.2895, 0.6250, 0.6587, 0.6754, 0.4062, 0.4277, 0.5256, 0.5652,
              0.6061, 0.6090, 0.7586, 0.8125, 0.4667, 0.7513]
}

# Convert data to pandas DataFrame
df_uncensored = pd.DataFrame(data_uncensored)
df_safety_tuned = pd.DataFrame(data_safety_tuned)

# Perform paired t-test
t_statistic, p_value = stats.ttest_rel(df_uncensored["Value"], df_safety_tuned["Value"])

# Set significance level
alpha = 0.05

# Print results
print("Paired t-test results:")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Interpret results
if p_value < alpha:
    print("There is a statistically significant difference in performance between the uncensored and safety-tuned versions.")
else:
    print("There is no statistically significant difference in performance between the uncensored and safety-tuned versions.")
