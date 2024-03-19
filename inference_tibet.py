import os
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
import argparse

SPECIAL_MARK = ["'", "+", "."]

def text_to_char(text: str):
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

    return phonemes

def get_text_token_id(text:str):
    with open("data/phonemes.json", "r") as fi:
        phoemes_name_to_id = json.load(fi)
    phonemes = text_to_char(text)
    phoneme_ids = [phoemes_name_to_id[phoneme] for phoneme in phonemes]

    return phoneme_ids
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--text", type=str, help="text to be synthesized")
    args.add_argument("--outputdir",default="./results", type=str, help="a folder saving synthesized wavs")
    args.add_argument("--checkpoint",default="", type=str, help="vits model checkpoint")
    args.add_argument("--config",default="configs/tibet_base.json", type=str)
    args.add_argument("--reference_audio",required=True, type=str)

    args = args.parse_args()

    ids = get_text_token_id(args.text)
    ids = torch.LongTensor(ids)

    hps = utils.get_hparams_from_file(args.config)
    device = torch.device("cuda:0")
    saved_dir = args.outputdir
    os.makedirs(saved_dir, exist_ok=True)

    net_g = SynthesizerTrn(
      40 if hps.data.Tibet else len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).to(device)

    net_g, *_ = utils.load_checkpoint(args.checkpoint, net_g, None)
    net_g.eval()

    wav_gst, _ = utils.load_wav_to_torch(args.reference_audio, sr=hps.data.sampling_rate)
    spec = spectrogram_torch(wav_gst.unsqueeze(0), hps.data.filter_length, 
                hps.data.sampling_rate, hps.data.hop_length,
                hps.data.win_length)
    with torch.no_grad():
        ids = ids.unsqueeze(0).to(device)
        spec = spec.to(device)
        length = torch.LongTensor([ids.shape[1]]).to(device)
        audio = net_g.infer(ids, length, spec, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        saved_path = os.path.join(saved_dir, "test.wav")
        audio = audio * 32768.0
        audio = audio.astype(np.int16)

        write(saved_path, hps.data.sampling_rate, audio)
