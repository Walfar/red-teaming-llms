import sys
sys.path.append('..')

from toxicity.Conversation import *
from utils import create_folders_if_not_exist

class ToxEval():
    def __init__(self,nb_rounds,redteam_model,target_model,is_preset=False,gen_policy="simple_generation",saved_challenges_file_name="saved_challenges"):
        self.nb_rounds = nb_rounds
        self.redteam_model = redteam_model
        self.target_model = target_model
        self.is_preset = is_preset
        self.all_challenges = {}
        self.all_results = {}
        self.gen_policy = gen_policy
        self.saved_challenges_file_name = saved_challenges_file_name

    def write_results(self):
        total_nb_successful_attacks_distilbert = 0
        total_nb_successful_attacks_roberta = 0 
        tox_scores = []
        dir_path = f"../results/toxicity/{self.target_model.name}"
        create_folders_if_not_exist(dir_path)
        path = dir_path+"/results.txt"
        with open(path,"w") as f:
            for conv_id, results in self.all_results.items():
                f.write(f"Results for conv_id {conv_id}:\n")
                f.write(f"Nb successful attacks distilbert : {results[0]}\n")
                f.write(f"Nb successful attacks roberta : {results[1]}\n")
                f.write(f"==============================\n")
                total_nb_successful_attacks_distilbert += results[0]
                total_nb_successful_attacks_roberta += results[1]
                tox_scores.append(results[2])
            f.write(f"Total results over all rounds:\n")
            f.write(f"Nb successful attacks distilbert : {total_nb_successful_attacks_distilbert}\n")
            f.write(f"Nb successful attacks roberta : {total_nb_successful_attacks_roberta}\n")
        
        result_df = pd.concat(tox_scores, ignore_index=True)
        result_df.to_csv(f"../results/toxicity/{self.target_model.name}/tox_scores.csv")

    def run_eval(self):
        if self.is_preset:
            self.load_challenges()
        for conv_id in range(self.nb_rounds):
            conv = Conversation(self.redteam_model,self.target_model,10,self.gen_policy,conv_id,with_history=True)
            if self.is_preset:
                prompts = self.all_challenges[conv_id]
                conv.preset_run(prompts)
            else:
                conv.run()
                self.all_challenges[conv_id] = conv.get_challenges()
            self.all_results[conv_id] = conv.get_results()
        self.write_results()
        if not self.is_preset:
            self.write_challenges()

    def write_challenges(self):
        create_folders_if_not_exist(f"../results/toxicity/{self.saved_challenges_file_name}")
        for conv_id, challenges in self.all_challenges.items():
            utils.write_prompts(f"../results/toxicity/{self.saved_challenges_file_name}",f"conv{conv_id}.txt",challenges)

    def load_challenges(self):
        for conv_id in range(self.nb_rounds):
            prompts = []
            with open(f"../results/toxicity/{self.saved_challenges_file_name}/conv{conv_id}.txt","r") as f:
                lines = f.readlines()
                current_prompt = ""
                for line in lines:
                    if line.startswith("prompt:"):
                        if current_prompt != "":
                            prompts.append(current_prompt)
                            current_prompt = ""
                        current_prompt += line[7:]
                    else:
                        current_prompt += line
                prompts.append(current_prompt)
            self.all_challenges[conv_id] = prompts

    
    

