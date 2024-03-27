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
import gradio
import re
from pypinyin import lazy_pinyin, Style

# lazy_pinyin('衣裳', style=Style.TONE3, neutral_tone_with_five=True)
punctunations = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'

with open("data/pinyin-lexicon-r.txt", "r") as fi:
    lines = [line.strip() for line in fi.readlines()]
pinyin_to_phoneme = {}
for line in lines:
    key = line.split()[0]
    value = " ".join(line.split()[1:])
    pinyin_to_phoneme[key] = value
    
with open("data/pinyin.json", "r") as fi:
        phoemes_name_to_id = json.load(fi)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
hps = utils.get_hparams_from_file("./configs/aishell3.json")
net_g = SynthesizerTrn(
      224, # aishell3 total phonemes numbers
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model).to(device)
net_g, *_ = utils.load_checkpoint("logs/aishell3/G_250000.pth", net_g, None)
net_g.eval()



def get_token_id(phonemes:str):
    
    phoneme_ids = [phoemes_name_to_id[phoneme] for phoneme in phonemes]

    return phoneme_ids
    
def text_to_phoneme(text: list):
    pinyins_list = []
    for item in text:
        pinyins_list.append(lazy_pinyin(item, style=Style.TONE3, neutral_tone_with_five=True, tone_sandhi=True))
    
    phonemes_list = []
    for pinyins in pinyins_list:
        phonemes_list.append([pinyin_to_phoneme[pinyin] for pinyin in pinyins])

    return phonemes_list

def text_to_speech(text: str, spk_id: int):
    
    text = re.sub('[{}]'.format(punctunations)," ",text)
    text = text.split()
    phonemes_list = text_to_phoneme(text)
    phonemes_sequence = []
    for index, phonemes in enumerate(phonemes_list):
        if index == 0:
            phonemes_sequence.append("sil " + " ".join(phonemes) + " sil")
        else:
            phonemes_sequence.append(" ".join(phonemes) + " sil")
    
    ids = get_token_id((" ".join(phonemes_sequence)).split())
    ids = torch.LongTensor(ids).unsqueeze(0).to(device)
    length = torch.LongTensor([ids.shape[1]]).to(device)
    sid = torch.LongTensor([spk_id]).to(device)
    audio = net_g.infer(ids, length, sid=sid, max_len=1000, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0].cpu().detach().float().numpy()
    audio = (audio * 32768.0).astype(np.int16)

    print(audio.shape)
    write("./results/test_aishell3.wav", hps.data.sampling_rate, audio[0,0])
    return "./results/test_aishell3.wav"







if __name__ == "__main__":

    web = gradio.Interface(fn=text_to_speech, inputs=["text", gradio.Dropdown(list(range(0,100)), label="speaker", info="可选择的说话人id")],
                     outputs="audio")
    web.launch()