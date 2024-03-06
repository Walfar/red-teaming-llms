""" This file is used to determine from the MMLU results what is the category with the highest accuracy and the one with the lowest for a given model"""

import re

# We paste here the printed results from running the evaluation harness on a specific LLM
data = """
|                      Task                       |Version| Metric |Value |   |Stderr|
|-------------------------------------------------|------:|--------|-----:|---|-----:|
|hendrycksTest-abstract_algebra                   |      1|acc     |0.3400|±  |0.0476|
|                                                 |       |acc_norm|0.3400|±  |0.0476|
|hendrycksTest-anatomy                            |      1|acc     |0.5111|±  |0.0432|
|                                                 |       |acc_norm|0.5111|±  |0.0432|
|hendrycksTest-astronomy                          |      1|acc     |0.5263|±  |0.0406|
|                                                 |       |acc_norm|0.5263|±  |0.0406|
|hendrycksTest-business_ethics                    |      1|acc     |0.4800|±  |0.0502|
|                                                 |       |acc_norm|0.4800|±  |0.0502|
|hendrycksTest-clinical_knowledge                 |      1|acc     |0.5811|±  |0.0304|
|                                                 |       |acc_norm|0.5811|±  |0.0304|
|hendrycksTest-college_biology                    |      1|acc     |0.5486|±  |0.0416|
|                                                 |       |acc_norm|0.5486|±  |0.0416|
|hendrycksTest-college_chemistry                  |      1|acc     |0.3800|±  |0.0488|
|                                                 |       |acc_norm|0.3800|±  |0.0488|
|hendrycksTest-college_computer_science           |      1|acc     |0.3700|±  |0.0485|
|                                                 |       |acc_norm|0.3700|±  |0.0485|
|hendrycksTest-college_mathematics                |      1|acc     |0.3800|±  |0.0488|
|                                                 |       |acc_norm|0.3800|±  |0.0488|
|hendrycksTest-college_medicine                   |      1|acc     |0.4509|±  |0.0379|
|                                                 |       |acc_norm|0.4509|±  |0.0379|
|hendrycksTest-college_physics                    |      1|acc     |0.2647|±  |0.0439|
|                                                 |       |acc_norm|0.2647|±  |0.0439|
|hendrycksTest-computer_security                  |      1|acc     |0.6600|±  |0.0476|
|                                                 |       |acc_norm|0.6600|±  |0.0476|
|hendrycksTest-conceptual_physics                 |      1|acc     |0.4128|±  |0.0322|
|                                                 |       |acc_norm|0.4128|±  |0.0322|
|hendrycksTest-econometrics                       |      1|acc     |0.2544|±  |0.0410|
|                                                 |       |acc_norm|0.2544|±  |0.0410|
|hendrycksTest-electrical_engineering             |      1|acc     |0.4897|±  |0.0417|
|                                                 |       |acc_norm|0.4897|±  |0.0417|
|hendrycksTest-elementary_mathematics             |      1|acc     |0.3042|±  |0.0237|
|                                                 |       |acc_norm|0.3042|±  |0.0237|
|hendrycksTest-formal_logic                       |      1|acc     |0.2778|±  |0.0401|
|                                                 |       |acc_norm|0.2778|±  |0.0401|
|hendrycksTest-global_facts                       |      1|acc     |0.3600|±  |0.0482|
|                                                 |       |acc_norm|0.3600|±  |0.0482|
|hendrycksTest-high_school_biology                |      1|acc     |0.6484|±  |0.0272|
|                                                 |       |acc_norm|0.6484|±  |0.0272|
|hendrycksTest-high_school_chemistry              |      1|acc     |0.4532|±  |0.0350|
|                                                 |       |acc_norm|0.4532|±  |0.0350|
|hendrycksTest-high_school_computer_science       |      1|acc     |0.5300|±  |0.0502|
|                                                 |       |acc_norm|0.5300|±  |0.0502|
|hendrycksTest-high_school_european_history       |      1|acc     |0.6424|±  |0.0374|
|                                                 |       |acc_norm|0.6424|±  |0.0374|
|hendrycksTest-high_school_geography              |      1|acc     |0.6263|±  |0.0345|
|                                                 |       |acc_norm|0.6263|±  |0.0345|
|hendrycksTest-high_school_government_and_politics|      1|acc     |0.7513|±  |0.0312|
|                                                 |       |acc_norm|0.7513|±  |0.0312|
|hendrycksTest-high_school_macroeconomics         |      1|acc     |0.4718|±  |0.0253|
|                                                 |       |acc_norm|0.4718|±  |0.0253|
|hendrycksTest-high_school_mathematics            |      1|acc     |0.2704|±  |0.0271|
|                                                 |       |acc_norm|0.2704|±  |0.0271|
|hendrycksTest-high_school_microeconomics         |      1|acc     |0.5252|±  |0.0324|
|                                                 |       |acc_norm|0.5252|±  |0.0324|
|hendrycksTest-high_school_physics                |      1|acc     |0.2980|±  |0.0373|
|                                                 |       |acc_norm|0.2980|±  |0.0373|
|hendrycksTest-high_school_psychology             |      1|acc     |0.6936|±  |0.0198|
|                                                 |       |acc_norm|0.6936|±  |0.0198|
|hendrycksTest-high_school_statistics             |      1|acc     |0.3704|±  |0.0329|
|                                                 |       |acc_norm|0.3704|±  |0.0329|
|hendrycksTest-high_school_us_history             |      1|acc     |0.7402|±  |0.0308|
|                                                 |       |acc_norm|0.7402|±  |0.0308|
|hendrycksTest-high_school_world_history          |      1|acc     |0.7257|±  |0.0290|
|                                                 |       |acc_norm|0.7257|±  |0.0290|
|hendrycksTest-human_aging                        |      1|acc     |0.6054|±  |0.0328|
|                                                 |       |acc_norm|0.6054|±  |0.0328|
|hendrycksTest-human_sexuality                    |      1|acc     |0.6870|±  |0.0407|
|                                                 |       |acc_norm|0.6870|±  |0.0407|
|hendrycksTest-international_law                  |      1|acc     |0.7190|±  |0.0410|
|                                                 |       |acc_norm|0.7190|±  |0.0410|
|hendrycksTest-jurisprudence                      |      1|acc     |0.6574|±  |0.0459|
|                                                 |       |acc_norm|0.6574|±  |0.0459|
|hendrycksTest-logical_fallacies                  |      1|acc     |0.6933|±  |0.0362|
|                                                 |       |acc_norm|0.6933|±  |0.0362|
|hendrycksTest-machine_learning                   |      1|acc     |0.3036|±  |0.0436|
|                                                 |       |acc_norm|0.3036|±  |0.0436|
|hendrycksTest-management                         |      1|acc     |0.6699|±  |0.0466|
|                                                 |       |acc_norm|0.6699|±  |0.0466|
|hendrycksTest-marketing                          |      1|acc     |0.7650|±  |0.0278|
|                                                 |       |acc_norm|0.7650|±  |0.0278|
|hendrycksTest-medical_genetics                   |      1|acc     |0.5100|±  |0.0502|
|                                                 |       |acc_norm|0.5100|±  |0.0502|
|hendrycksTest-miscellaneous                      |      1|acc     |0.7369|±  |0.0157|
|                                                 |       |acc_norm|0.7369|±  |0.0157|
|hendrycksTest-moral_disputes                     |      1|acc     |0.5838|±  |0.0265|
|                                                 |       |acc_norm|0.5838|±  |0.0265|
|hendrycksTest-moral_scenarios                    |      1|acc     |0.2469|±  |0.0144|
|                                                 |       |acc_norm|0.2469|±  |0.0144|
|hendrycksTest-nutrition                          |      1|acc     |0.6111|±  |0.0279|
|                                                 |       |acc_norm|0.6111|±  |0.0279|
|hendrycksTest-philosophy                         |      1|acc     |0.6463|±  |0.0272|
|                                                 |       |acc_norm|0.6463|±  |0.0272|
|hendrycksTest-prehistory                         |      1|acc     |0.6389|±  |0.0267|
|                                                 |       |acc_norm|0.6389|±  |0.0267|
|hendrycksTest-professional_accounting            |      1|acc     |0.4007|±  |0.0292|
|                                                 |       |acc_norm|0.4007|±  |0.0292|
|hendrycksTest-professional_law                   |      1|acc     |0.4224|±  |0.0126|
|                                                 |       |acc_norm|0.4224|±  |0.0126|
|hendrycksTest-professional_medicine              |      1|acc     |0.5147|±  |0.0304|
|                                                 |       |acc_norm|0.5147|±  |0.0304|
|hendrycksTest-professional_psychology            |      1|acc     |0.5359|±  |0.0202|
|                                                 |       |acc_norm|0.5359|±  |0.0202|
|hendrycksTest-public_relations                   |      1|acc     |0.6000|±  |0.0469|
|                                                 |       |acc_norm|0.6000|±  |0.0469|
|hendrycksTest-security_studies                   |      1|acc     |0.6082|±  |0.0313|
|                                                 |       |acc_norm|0.6082|±  |0.0313|
|hendrycksTest-sociology                          |      1|acc     |0.7612|±  |0.0301|
|                                                 |       |acc_norm|0.7612|±  |0.0301|
|hendrycksTest-us_foreign_policy                  |      1|acc     |0.8200|±  |0.0386|
|                                                 |       |acc_norm|0.8200|±  |0.0386|
|hendrycksTest-virology                           |      1|acc     |0.4518|±  |0.0387|
|                                                 |       |acc_norm|0.4518|±  |0.0387|
|hendrycksTest-world_religions                    |      1|acc     |0.7135|±  |0.0347|
|                                                 |       |acc_norm|0.7135|±  |0.0347|
"""

# Regular expression pattern to match numerical values
pattern = r'\d+\.\d+'

# Initialize variables for min and max values
min_value = float('inf')
max_value = float('-inf')
min_category = ""
max_category = ""

# Split data into lines and iterate through each line
for line in data.split('\n'):
    # Search for numerical values in the line
    matches = re.findall(pattern, line)
    # If numerical values are found
    if matches:
        # Extract the numerical value as a float
        value = float(matches[0])
        # Extract the category
        category = line.split('|')[1].strip()
        # Update min and max values and corresponding categories
        if value < min_value:
            min_value = value
            min_category = category
        if value > max_value:
            max_value = value
            max_category = category

# Print the min and max values and their corresponding categories
print("Minimum Value:", min_value, "Category:", min_category)
print("Maximum Value:", max_value, "Category:", max_category)