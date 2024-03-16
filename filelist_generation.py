import os
import json
from pathlib import Path
import argparse
SPECIAL_MARK = ["'", "+", "."]

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--input_data_path",type=str)
    args.add_argument("--outputdir",type=str)
    args = args.parse_args()
    data_path = args.input_data_path
    data_path = Path(data_path)

    output_dir = Path(args.outputdir)

    file_list = []
    for file in data_path.rglob("*.txt"):
        with open(file, "r") as fi:
            lines = [line.strip() for line in fi.readlines()]
        text = lines[0]

        phonemes = []
        for item in text.split():
            
            index = 0
            while index < len(item):
                if index != len(item) - 1:
                    if item[index + 1] in SPECIAL_MARK or item[index] == "'":
                        phoneme = item[index] + item[index+1] 
                        index += 2
                    else:
                        phoneme = item[index]
                        index += 1
                else:
                    phoneme = item[index]
                    index += 1
                
                phonemes.append(phoneme)
        # 开头和结尾插入sil（静音段音素）
        phonemes.insert(0,"sil")
        phonemes.append("sil")
        wav_path = file.parent/(file.stem + ".wav")
        assert wav_path.exists()
        line = str(wav_path) + "|" + " ".join(phonemes)
        file_list.append(line)
            
    trainning_length = int(0.95 * len(file_list))
    with open(output_dir/"tibet_train.txt", "w") as fi:
        for line in file_list[:trainning_length]:
            fi.write(line + "\n")
    
    with open(output_dir/"tibet_valid.txt", "w") as fi:
        for line in file_list[trainning_length:]:
            fi.write(line + "\n")

