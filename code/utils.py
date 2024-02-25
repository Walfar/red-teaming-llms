import os
import pickle

def create_folders_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def write_prompts(dir_path,filename,prompts,and_response=False,mode="w"):
        create_folders_if_not_exist(dir_path)
        with open(dir_path+"/"+filename,mode) as f:
            for prompt in prompts:
                if not and_response:
                    f.write("prompt: " + prompt + "\n")
                    f.write("===================================\n")
                else:
                    f.write("prompt: " + prompt[0] + "\n")
                    f.write("response: " + prompt[1] + "\n")
                    f.write("===================================\n")
