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

    phonemes = dict()
    phonemes_id = 0

    for file in data_path.rglob("*.txt"):
        with open(file, "r") as fi:
            lines = [line.strip() for line in fi.readlines()]
        text = lines[0]

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
                
                if phonemes.get(phoneme) == None:
                    phonemes[phoneme] = phonemes_id
                    phonemes_id += 1
    
    print("number of phonemes: ", len(phonemes.keys()))
    with open(output_dir/"phonemes.json", "w") as fi:
        json.dump(phonemes, fi, indent=4)

