import os
import json
from pathlib import Path

if __name__ == "__main__":

    data_path = "/home/wangkengxue/code2/vits/data/tibet_data"
    data_path = Path(data_path)

    phonemes_path = "/home/wangkengxue/code2/vits/data/phonemes.json"
    with open(phonemes_path,"r") as fi:
        phonemes = json.load(fi)

    for file in data_path.rglob("*.txt"):
        with open(file, "r") as fi:
            lines = [line.strip() for line in fi.readlines()]
        text = lines[0]

        for item in text.split():
            index = 0
            while index < len(item):
                if index != len(item) - 1:
                    if item[index + 1] == "'" or item[index + 1] == ".":
                        phoneme = item[index] + item[index+1] 
                        index += 2
                    else:
                        phoneme = item[index]
                        index += 1
                else:
                    phoneme = item[index]
                    index += 1
                
                if not phonemes.get(phoneme):
                    print(phoneme,"|",file.stem)
    
