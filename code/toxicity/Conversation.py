import sys
sys.path.append('..')

from Model import ToxicDistilbert, ToxicRoberta, GenerativeModel,RedTeamModel
import pandas as pd
import utils as utils

class Conversation:

    def __init__(self,redteam_model:RedTeamModel,target_model:GenerativeModel,nb_turns,gen_policy,conv_id,with_history=True):

        self.redteam_model = redteam_model
        self.target_model = target_model
        self.nb_turns = nb_turns
        self.gen_policy = gen_policy
        self.conv_id = 0

        self.toxic_distilbert = ToxicDistilbert(0)
        self.toxic_roberta = ToxicRoberta(0)

        self.successful_attacks_distilbert = []
        self.successful_attacks_roberta = []

        self.conv_id = conv_id

        self.tox_scores = pd.DataFrame(columns=['conversation id','turn','score distilbert', 'score roberta'])
        self.last_challenge = None
        self.last_response = None

        self.records = []
        self.keep_going = True
        self.with_history = with_history

    def generate_challenge(self):

        input_score_distilbert = None
        input_score_roberta = None

        if self.gen_policy == "simple_generation":

            self.last_challenge = self.redteam_model.generate_single(self.last_response)
            _, input_score_distilbert = self.toxic_distilbert.evaluate(self.last_challenge)
            _, input_score_roberta = self.toxic_roberta.evaluate(self.last_challenge)

        elif self.gen_policy == "first_tox": #efficient against uncensored models (upper bounded by 10 rounds of gen)
            for count in range(10):
                challenge = self.redteam_model.generate_single(self.last_response)
                tox_label_distilbert, score_distilbert = self.toxic_distilbert.evaluate(challenge)
                tox_label_roberta, score_roberta = self.toxic_roberta.evaluate(challenge)
                if tox_label_distilbert and tox_label_roberta:
                    count = 9
                if count == 9:
                    self.last_challenge = challenge
                    input_score_distilbert = score_distilbert
                    input_score_roberta = score_roberta

        elif self.gen_policy == "best_tox": #efficient against uncensored models (10 rounds of gen)
            
            max_tox_resp = None
            max_score = 0
            for _ in range(10):

                challenge = self.redteam_model.generate_single(self.last_response)
                if max_tox_resp == None:
                    max_tox_resp = challenge
                    _, max_score = self.toxic_distilbert.evaluate(challenge)
                else:
                    tox_label_distilbert, score_distilbert = self.toxic_distilbert.evaluate(challenge)
                    if score_distilbert > max_score:
                        max_score = score_distilbert
                        max_tox_resp = challenge

            self.last_challenge = max_tox_resp
            input_score_distilbert = max_score
            _, input_score_roberta = self.toxic_roberta.evaluate(max_tox_resp)

        else:
            raise(f"Generation Policy Unknown {self.gen_policy}")
        
        return input_score_distilbert, input_score_roberta
    
    def process_response(self,response):
        return response.replace('"', '')
    
    def evaluate_response(self):

        attack_success, output_score_distilbert = self.toxic_distilbert.evaluate(self.last_response)
        if attack_success:
            self.successful_attacks_distilbert.append((self.last_challenge,self.last_response))
        attack_success, output_score_roberta = self.toxic_roberta.evaluate(self.last_response)
        if attack_success:
            self.successful_attacks_roberta.append((self.last_challenge,self.last_response))
        return output_score_distilbert, output_score_roberta
    
    def get_challenges(self):
        return self.records

    # in this run, we used recorded prompts and don't record toxicity scores
    def preset_run(self,prompts):
        self.target_model.reset_chat()
        responses = self.target_model.generate_multiple(prompts)
        if self.with_history:
            utils.write_prompts(f"../results/toxicity/{self.target_model.name}/conv{self.conv_id}","history.txt",zip(prompts,responses),and_response=True)

        for prompt, response in zip(prompts,responses):
            self.last_response = response
            self.last_challenge = prompt
            self.evaluate_response()

        utils.write_prompts(f"../results/toxicity/{self.target_model.name}/conv{self.conv_id}","successful_attacks_distilbert.txt",self.successful_attacks_distilbert,and_response=True)
        utils.write_prompts(f"../results/toxicity/{self.target_model.name}/conv{self.conv_id}","successful_attacks_roberta.txt",self.successful_attacks_roberta,and_response=True)
        
    def run(self):
        prompts = []
        responses = []
        for turn in range(self.nb_turns):
            self.target_model.reset_chat()
            input_score_distilbert, input_score_roberta = self.generate_challenge()
            response = self.target_model.generate_single(self.last_challenge)

            if not len(response):
                self.keep_going = False
            if self.last_response == response:
                self.keep_going = False
            #self.last_response = response.replace("\n", " ").strip() #interesting.... why?

            #self.last_response = self.process_response(response)
            self.last_response = response
            
            output_score_distilbert, output_score_roberta = self.evaluate_response()

            self.records.append(self.last_challenge)

            new_row = {'conversation id':self.conv_id,'turn':turn,'score distilbert':(input_score_distilbert,output_score_distilbert), 'score roberta':(input_score_roberta,output_score_roberta)}
            self.tox_scores = pd.concat([self.tox_scores, pd.DataFrame([new_row])], ignore_index=True)
            
            prompts.append(self.last_challenge)
            responses.append(self.last_response)
            
        if self.with_history:
            utils.write_prompts(f"../results/toxicity/{self.target_model.name}/conv{self.conv_id}","history.txt",zip(prompts,responses),and_response=True)
            
        utils.write_prompts(f"../results/toxicity/{self.target_model.name}/conv{self.conv_id}","successful_attacks_distilbert.txt",self.successful_attacks_distilbert,and_response=True)
        utils.write_prompts(f"../results/toxicity/{self.target_model.name}/conv{self.conv_id}","successful_attacks_roberta.txt",self.successful_attacks_roberta,and_response=True)
        
        
    def get_results(self):
        nb_successful_attacks_distilbert = len(self.successful_attacks_distilbert)
        nb_successful_attacks_roberta = len(self.successful_attacks_roberta)
        return nb_successful_attacks_distilbert, nb_successful_attacks_roberta, self.tox_scores
        