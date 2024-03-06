import re
import os

def compute_unique(pairs):
    unique_pairs = {}
    for pair in pairs:
            prompt_match = re.search(r'prompt:\s*(.*?)\s*response:', pair, re.DOTALL)
            if prompt_match:
                prompt = prompt_match.group(1).strip()
                if prompt not in unique_pairs:
                    unique_pairs[prompt] = pair
    return list(unique_pairs.values())

def filter_complied_attempts(categ, jailbreak_name, model_name, total_nb_rps, total_nb_jailbreaks, total_nb_toxic):

    file_path = f'../results/compliance/jailbreaks/{model_name}/complied/{jailbreak_name}_{categ}.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r') as input_file:
            data = input_file.read()

        complied_pairs = [section.strip() for section in data.split("===================================") if section.strip()]
        complied_pairs = compute_unique(complied_pairs)

        rejected_pairs_size = 0
        rejected_path = f'../results/compliance/jailbreaks/{model_name}/rejected/{jailbreak_name}_{categ}.txt'
        if os.path.exists(rejected_path):
            with open(file_path, 'r') as input_file:
                data = input_file.read()
            rejected_pairs = [section.strip() for section in data.split("===================================") if section.strip()]
            rejected_pairs = compute_unique(rejected_pairs)
            rejected_pairs_size = len(rejected_pairs)
        
        
        #print(f"Number of successful role-plays for {model_name}, {categ}: {len(pairs)}")
        #print(f"Number of successful jailbreaks for {model_name}, {categ}: {len(filtered_pairs)}")
        
        toxic_path = f'../results/compliance/jailbreaks/{model_name}/toxic/{jailbreak_name}_{categ}.txt'
        toxic_pairs_size = 0
        if os.path.exists(toxic_path):
            with open(toxic_path, 'r') as input_file:
                data = input_file.read()
            toxic_pairs = [section.strip() for section in data.split("===================================") if section.strip()]
            toxic_pairs = compute_unique(toxic_pairs)
            toxic_pairs_size = len(toxic_pairs)
            #print(f"Number of toxic responses for {model_name}, {categ}: {toxic_pairs_size}")
        
        return total_nb_rps+len(complied_pairs)+rejected_pairs_size, total_nb_jailbreaks+len(complied_pairs), total_nb_toxic+toxic_pairs_size
    else:
        #print(f"File not found for category {categ}, counting as 0.")
        return total_nb_rps, total_nb_jailbreaks, total_nb_toxic

def main(jailbreak_name,model_name):
    total_nb_rps = 0
    total_nb_jailbreaks = 0
    total_nb_toxic = 0
    print("========================")
    print(jailbreak_name)

    categories = ["Eco","Fin","Hat","Hea","Ill","Phy","Pri","Por","Pol","Mal","Fra","Leg","Gov"]
    for category in categories: 
        total_nb_rps, total_nb_jailbreaks, total_nb_toxic = filter_complied_attempts(category,jailbreak_name,model_name,total_nb_rps,total_nb_jailbreaks,total_nb_toxic)

    print(f"Total number of successful role-plays for {model_name}: {total_nb_rps}")
    print(f"Total number of successful jailbreaks for {model_name}: {total_nb_jailbreaks}")
    print(f"Total number of toxic responses for {model_name}: {total_nb_toxic}")


main("apophis","llama2_13b_chat_uncensored")
main("betterdan","llama2_13b_chat_uncensored")
main("maximum","llama2_13b_chat_uncensored")

main("apophis","llama2_13b_chat")
main("betterdan","llama2_13b_chat")
main("maximum","llama2_13b_chat")

main("apophis","llama2_7b_chat")
main("betterdan","llama2_7b_chat")
main("maximum","llama2_7b_chat")

main("apophis","llama2_7b_chat_uncensored")
main("betterdan","llama2_7b_chat_uncensored")
main("maximum","llama2_7b_chat_uncensored")