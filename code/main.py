import argparse
import yaml
import subprocess

from Model import RedTeamModel, Llama2Chat7B, Llama2Chat13B, Llama2ChatUncensored7B, Llama2ChatUncensored13B, Llama2Chat7BGPTQ, Llama2Chat7BGGUF
from toxicity.ToxicityEval import ToxEval
from compliance.ComplianceEval import ComplianceEval


def choose_target_model(target_model_name,gpu_id):
    if target_model_name == "llama2_7b_chat":
        return Llama2Chat7B(gpu_id)
    elif target_model_name == "llama2_7b_chat_uncensored":
        return Llama2ChatUncensored7B(gpu_id)
    elif target_model_name == "llama2_13b_chat_uncensored":
        return Llama2ChatUncensored13B(gpu_id)
    elif target_model_name == "llama2_13b_chat":
        return Llama2Chat13B(gpu_id)
    elif target_model_name == "llama2_7b_chat_gptq":
        return Llama2Chat7BGPTQ(gpu_id)
    elif target_model_name == "llama2_7b_chat_gguf":
        return Llama2Chat7BGGUF(gpu_id)
    else:
        raise(f"Model name doesn't exist {target_model_name}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate toxicity through automatic red teaming, compliance with harmful datasets and performance with benchmarks")

    parser.add_argument("evaluation_type", choices=["toxicity", "compliance", "performance","jailbreak"], help="Type of evaluation")

    args = parser.parse_args()

    with open("config.yml", 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)


    if args.evaluation_type == "toxicity":

        toxicity_config = config.get('toxicity_config', {})

        if toxicity_config.get("eval_type") == "full":
            redteam_model = RedTeamModel(0)
            reference_model_name = toxicity_config.get("reference_model")

            target_model_names = toxicity_config.get("target_models")
            gpu_ids = toxicity_config.get("gpu_ids")
            if len(gpu_ids) != len(target_model_names) + 1:
                raise("There should be the same number of gpu ids as models (including reference model)")

            gen_policy = toxicity_config.get("gen_policy")

            target_models = []
            reference_model = choose_target_model(reference_model_name,gpu_ids[0])
            for gpu_id, target_model_name in zip(gpu_ids[1:],target_model_names):
                target_models.append(choose_target_model(target_model_name,gpu_id))

            initial_eval = ToxEval(20,redteam_model,reference_model,gen_policy=gen_policy)
            initial_eval.run_eval()
            for target_model in target_models:
                    eval = ToxEval(20,redteam_model,target_model,is_preset=True)
                    eval.run_eval()

    elif args.evaluation_type == "compliance":

        compliance_config = config.get('compliance_config', {})
        eval_type = compliance_config.get("eval_type")
        harmful_dataset_path = compliance_config.get("harmful_dataset_path")

        target_model_names = compliance_config.get("target_models")
        gpu_ids = compliance_config.get("gpu_ids")
        if len(gpu_ids) != len(target_model_names):
                raise("There should be the same number of gpu ids as models")
        target_models = []
        for gpu_id, target_model_name in zip(gpu_ids,target_model_names):
                target_models.append(choose_target_model(target_model_name,gpu_id))

        for target_model in target_models:
            print(f"Compliance eval for: {target_model.name}")
            compliance_eval = ComplianceEval(harmful_dataset_path, eval_type, target_model)
            compliance_eval.run()

    elif args.evaluation_type == "jailbreak":

        jailbreaks_config = config.get('jailbreaks_config', {})  
        target_model_names = jailbreaks_config.get("target_models")
        gpu_ids = jailbreaks_config.get("gpu_ids")

        if len(gpu_ids) != len(target_model_names):
                raise("There should be the same number of gpu ids as models")
        target_models = []
        for gpu_id, target_model_name in zip(gpu_ids,target_model_names):
                target_models.append(choose_target_model(target_model_name,gpu_id))
        
        for target_model in target_models:
            compliance_eval = ComplianceEval("../data/compliance/harmful_dataset.csv", "jailbreak", target_model)
            compliance_eval.eval_jailbreaks()

    elif args.evaluation_type == "performance":

        performance_config = config.get('performance_config', {})  
        target_model_names = performance_config.get("target_models")
        benchmarks = performance_config.get("benchmarks")

        target_model_paths = {
            "llama2_7b_chat": "meta-llama/Llama-2-7b-chat-hf",
            "llama2_7b_chat_uncensored": "georgesung/llama2_7b_chat_uncensored",
            "llama2_13b_chat_uncensored": "arogov/llama2_13b_chat_uncensored",
            "llama2_13b_chat": "meta-llama/Llama-2-13b-chat-hf"
        }

        for target_model_name in target_model_names:
            target_model_path = target_model_paths[target_model_name]
            if "mmlu" in benchmarks:
                subprocess.run(["python3", "performance/lm-evaluation-harness/main.py", "--model", "hf-causal", "--model_args", f"pretrained={target_model_path},use_accelerate=True ", "--tasks", "hendrycksTest*", "--device", "cuda:0"])
            if "mnli" in benchmarks:
                subprocess.run(["python3", "performance/lm-evaluation-harness/main.py", "--model", "hf-causal", "--model_args", f"pretrained={target_model_path},use_accelerate=True ", "--tasks", "mnli*", "--device", "cuda:0"])
            if "ethics" in benchmarks:
                subprocess.run(["python3", "performance/lm-evaluation-harness/main.py", "--model", "hf-causal", "--model_args", f"pretrained={target_model_path},use_accelerate=True ", "--tasks", "ethics_*", "--device", "cuda:0"])
            if "truthfulqa" in benchmarks:
                subprocess.run(["python3", "performance/lm-evaluation-harness/main.py", "--model", "hf-causal", "--model_args", f"pretrained={target_model_path},use_accelerate=True ", "--tasks", "truthfulqa_mc", "--device", "cuda:0"])
            if "hellaswag" in benchmarks:
                subprocess.run(["python3", "performance/lm-evaluation-harness/main.py", "--model", "hf-causal", "--model_args", f"pretrained={target_model_path},use_accelerate=True ", "--tasks", "hellaswag", "--num_fewshot=10", "--device", "cuda:0"])
            if "ARC" in benchmarks:
                subprocess.run(["python3", "performance/lm-evaluation-harness/main.py", "--model", "hf-causal", "--model_args", f"pretrained={target_model_path},use_accelerate=True ", "--tasks", "arc_challenge", "--num_fewshot=25", "--device", "cuda:0"])


    else:
        print("Invalid evaluation type. Please choose 'toxicity', 'compliance' or 'performance' or 'jailbreak'.")
        return


if __name__ == "__main__":
    main()

