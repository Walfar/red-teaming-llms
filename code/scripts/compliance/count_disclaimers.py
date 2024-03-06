import os

def do(model_name):
    print(model_name)
    dir = f"../results/compliance/{model_name}/"
    folders = [dir+"complied_attempts", dir+"complied_attempts_with_disclaimer", dir+"rejected_attempts"]
    files = ["attempts_Fin.txt", "attempts_Hea.txt", "attempts_Leg.txt"]

    for folder in folders:
        print(f"Folder: {folder}")
        for file_name in files:
            file_path = os.path.join(folder, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    count = sum(1 for line in file if line.strip() == "===================================")
                print(f"{file_name}: {count} occurrences")
        print()

do("llama2_7b_chat")
do("llama2_7b_chat_uncensored")
do("llama2_13b_chat")
do("llama2_13b_chat_uncensored")