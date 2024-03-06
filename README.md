# Red Teaming Llms

## Project

This is the code used to evaluate toxicity, compliance to answer harmful questions and performance of the models Llama2 7B Chat, Llama2 13B Chat, Llama2 7B Chat Uncensored and Llama2 13B Chat Uncensored for the project "Red Teaming Uncensored and Sadfety-Tuned Large Language Models, exploring the trade-offs between utility and harmfulness"

## How to Run

1. Edit Configuration File

Before running the code, you need to edit the config.yml file to suit your preferences. Open the file and locate the toxicity_config, compliance_config, jailbreaks_config, or performance_config sections depending on the evaluation you want to perform. Modify the parameters according to your requirements. Here's what you need to adjust:

```yaml
toxicity_config:
  eval_type: full
  reference_model: llama2_7b_chat
  num_conversations: 20 
  nb_turns: 10
  target_models:
    - llama2_7b_chat_uncensored
    - llama2_13b_chat
    - llama2_13b_chat_uncensored
  gpu_ids:
    - 0
    - 0
    - 1
    - 2
  preset_prompts_path: ../data/toxicity/saved_challenges
  gen_policy: best_tox
```

2. Specify GPU IDs:

If you're using multiple GPUs, make sure to specify the GPU ID for each target model. Ensure that each model runs on a different GPU to prevent CUDA out-of-memory errors. For example, in the provided configuration, different GPU IDs are assigned to each target model:

```yaml
gpu_ids:
  - 0
  - 0
  - 1
  - 2
```

Adjust the GPU IDs according to your setup. In this setup, two 7B models can run on a single A100 GPU, while each 13B model requires a separate A100 GPU.

If you're utilizing multiple GPUs, allocate unique IDs for each GPU. For evaluations on all four models as mentioned, a total of 3 A100 GPUs were used.

Note: Ensure that your system has the necessary hardware requirements to accommodate multiple GPUs.

3. Run with Command Line:

Execute the code using the following command:

`python3 main.py [arg]`

Replace [arg] with the specific configuration you want to use (jailbreak, toxicity, compliance, or performance).

