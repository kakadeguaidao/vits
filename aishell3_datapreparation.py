import os
from pathlib import Path
import argparse
import tgt
import tqdm
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--train_wavs_dir", type=str, required=True)
    args.add_argument("--valid_wavs_dir", type=str, required=True)
    args.add_argument("--textgrid_dir", type=str, required=True)
    args.add_argument("--saved_dir", type=str, required=True)
    args.add_argument("--speakers_json", type=str, required=True)

    args = args.parse_args()

    train_wavs_dir = Path(args.train_wavs_dir)
    valid_wavs_dir = Path(args.valid_wavs_dir)
    textgrid_dir = args.textgrid_dir
    saved_dir = Path(args.saved_dir)
    with open(args.speakers_json, "r") as fi:
        speakers = json.load(fi)

    lines_train_to_write = []
    lines_valid_to_write = []
    for speaker in tqdm.tqdm(sorted(os.listdir(textgrid_dir))):
        for file in Path(os.path.join(textgrid_dir, speaker)).rglob("*.TextGrid"):
            phonemes = []
            textgrid = tgt.io.read_textgrid(file)
            textgrid_objs = textgrid.get_tier_by_name("phones")
            for _obj in textgrid_objs._objects:
                phoneme = _obj.text
                phonemes.append(phoneme)
            wav_file_name = file.stem + ".wav"
            if (train_wavs_dir/speaker/wav_file_name).exists():
                wav_file_path = train_wavs_dir/speaker/wav_file_name
                spk_indx = speakers[speaker]
                line = str(wav_file_path) + "|" + str(spk_indx) + "|" + " ".join(phonemes)
                lines_train_to_write.append(line)
            else:
                wav_file_path = valid_wavs_dir/speaker/wav_file_name
                spk_indx = speakers[speaker]
                line = str(wav_file_path) + "|" + str(spk_indx) + "|" + " ".join(phonemes)
                lines_valid_to_write.append(line)

    
    with open(saved_dir/"aishell3_train.txt","w") as fi:
        for line in lines_train_to_write:
            fi.write(line + "\n")
    
    with open(saved_dir/"aishell3_valid.txt","w") as fi:
        for line in lines_valid_to_write:
            fi.write(line + "\n")
