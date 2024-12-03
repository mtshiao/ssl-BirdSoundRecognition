import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import to_2tuple
import sys
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed, interpolate_pos_embed_audio, interpolate_patch_embed_audio, interpolate_pos_embed_img2audio
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from src import models_vit
from src.engine_finetune import train_one_epoch, evaluate, val_one_epoch
from src.dataset import AudiosetDataset, DistributedWeightedSampler, DistributedSamplerWrapper
from timm.models.vision_transformer import PatchEmbed
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Sampler
from pathlib import Path
import librosa
import glob
from tqdm import tqdm
import torch.nn.functional as F
import csv, os, sys
import json
import torchaudio
import numpy as np
import torch.nn.functional
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
import torch.distributed as dist
import random
import math
import psycopg2
import pandas as pd
from datetime import datetime
from collections import defaultdict

convert_to_mono_switch = True

script_dir = Path(__file__).resolve().parent
audio_dir_setting = script_dir /"inference_Audio"
audio_mono_dir_setting = script_dir / "temporary_file"
output_path = script_dir / "output"/ "infer_results"
model_pth_path = script_dir / "model" / "model_01_1_1_1.pth"
thresholds_filepath = script_dir / "Settings" / "optimal_f05_thresholds.txt"
label_mapping_path = script_dir / "Settings" / "finetune_labels_indices.csv"

def convert_to_mono(filepath, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = torchaudio.load(filepath)
    if audio.shape[0] == 2:
        mono_audio = torch.mean(audio, dim=0, keepdim=True)
        torchaudio.save(output_path, mono_audio, sample_rate)
    else:
        torchaudio.save(output_path, audio, sample_rate)

class InferenceDataset(Dataset):
    def __init__(self, segments, audio_conf, use_fbank=False, fbank_dir=None):
        self.segments = segments
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.frame_shift_ms = self.audio_conf.get('frame_shift_ms')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.low_freq = self.audio_conf.get('low_freq')
        self.high_freq = self.audio_conf.get('high_freq')

    def _wav2fbank(self, filename, start_time=None, end_time=None):
        waveform, sr = torchaudio.load(filename)
        if start_time is not None or end_time is not None:
            start_sample = int(start_time * sr) if start_time else 0
            end_sample = int(end_time * sr) if end_time else None
            waveform = waveform[:, start_sample:end_sample]
        waveform = waveform - waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform,
                                                htk_compat=True,
                                                sample_frequency=sr,
                                                use_energy=False,
                                                window_type='hanning',
                                                num_mel_bins=self.melbins,
                                                dither=0.0,
                                                frame_shift=self.frame_shift_ms,
                                                low_freq=self.low_freq,
                                                high_freq=self.high_freq)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        use_interpolation = True  
        if p != 0:
            if use_interpolation:
                fbank = fbank.unsqueeze(0).unsqueeze(0)
                fbank = F.interpolate(fbank, size=(target_length, self.melbins), mode='nearest')
                fbank = fbank.squeeze(0).squeeze(0)
            else:
                if p > 0:
                    m = torch.nn.ZeroPad2d((0, 0, 0, p))
                    fbank = m(fbank)
                elif p < 0:
                    fbank = fbank[:target_length, :]

        return fbank

    def _fbank(self, filename, start_time=None, end_time=None):
        fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
        fbank = np.load(fn1)
        if start_time is not None or end_time is not None:
            start_frame = int(start_time * 1000 / self.frame_shift_ms) if start_time else 0
            end_frame = int(end_time * 1000 / self.frame_shift_ms) if end_time else None
            fbank = fbank[start_frame:end_frame, :]

        return torch.from_numpy(fbank)

    def __getitem__(self, index):
        segment = self.segments[index]
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', None)

        if not self.use_fbank:
            fbank = self._wav2fbank(segment['wav'], start_time, end_time)
        else:
            fbank = self._fbank(segment['wav'], start_time, end_time)

        fbank = fbank.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        return fbank.unsqueeze(0), segment['wav']

    def __len__(self):
        return len(self.segments)

def get_audio_duration(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sample_rate
    return duration

def load_input_data(audio_path, audio_conf):
    audio_duration = get_audio_duration(audio_path)
    segments = generate_segments(audio_path, audio_duration)

    dataset = InferenceDataset(segments=segments, audio_conf=audio_conf)
    return dataset

def generate_segments(audio_path, audio_duration, segment_duration=1, gap=0.25):
    num_segments = int(np.ceil((audio_duration - segment_duration) / gap))
    segments = []

    for i in range(num_segments):
        start_time = i * gap
        end_time = start_time + segment_duration
        segments.append({
            "wav": audio_path,
            "start_time": start_time,
            "end_time": end_time
        })

    return segments

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

def inference(model, input_data, device):
    model.eval()
    input_data = input_data.to(device)

    with torch.no_grad():
        output = model(input_data)

    # return output.cpu().numpy()
    probabilities = torch.sigmoid(output)
    return probabilities.cpu().numpy()

def load_thresholds(filepath):
    thresholds = {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            thresholds[f'class_{i}'] = float(line.strip())
    return thresholds

thresholds = load_thresholds(thresholds_filepath)

label_mapping_df = pd.read_csv(label_mapping_path)

label_mapping = dict(zip(label_mapping_df['index'], label_mapping_df['display_name']))

def get_peak_info(row, thresholds, default_threshold, num_labels):
    peak_labels = []
    peak_probabilities = []
    for i in range(num_labels):
        prob = row[f'Label: {i}']
        display_name = label_mapping.get(i, f'Label: {i}')
        if prob >= thresholds.get(f'class_{i}', default_threshold):
            peak_labels.append(display_name)
            peak_probabilities.append(str(prob))
    return ', '.join(peak_labels), ', '.join(peak_probabilities)

ignore_label = 'NoneOfTheAbove'

use_time_interval_filter = True

time_interval = 2

def is_within_interval(start_time, end_time, interval):
    interval_start = int(start_time // interval) * interval
    interval_end = interval_start + interval
    return start_time >= interval_start and end_time <= interval_end

def group_by_interval(df, interval):
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        interval_start = int(float(row['start_time']) // interval) * interval
        interval_end = interval_start + interval
        if is_within_interval(float(row['start_time']), float(row['end_time']), interval):
            grouped[(interval_start, interval_end)].append(row)
    return grouped

def merge_groups(groups):
    merged_rows = []
    for (start_time, end_time), rows in groups.items():
        if not rows:
            continue
        peak_label_dict = defaultdict(list)
        peak_probability_dict = defaultdict(float)

        for row in rows:
            labels = row['peak_label'].split(', ')
            probabilities = row['peak_probability'].split(', ')

            for label, probability in zip(labels, probabilities):
                if probability:  # Check if probability is not an empty string
                    prob = float(probability)
                    if prob > peak_probability_dict[label]:
                        peak_probability_dict[label] = prob

        peak_labels = []
        peak_probabilities = []
        for label, prob in peak_probability_dict.items():
            peak_labels.append(label)
            peak_probabilities.append(f"{prob:.8f}")

        merged_row = {
            'created_at': rows[0]['created_at'],
            'audio_path': rows[0]['audio_path'],
            'start_time': start_time,
            'end_time': min(end_time, max(float(row['end_time']) for row in rows)),
            'peak_label': ', '.join(peak_labels),
            'peak_probability': ', '.join(peak_probabilities)
        }
        merged_rows.append(merged_row)
    return merged_rows

if convert_to_mono_switch:
    audio_dir = audio_dir_setting
else:
    audio_dir = audio_mono_dir_setting

audio_conf = {
    'num_mel_bins': 128,
    'target_length': 128,
    'freqm': 0,
    'timem': 0,
    'mixup': 0,
    'dataset': 'birdsong',
    'mode': 'inference',
    'mean': -7.535187,
    'std': 2.8788178,
    'noise': False,
    'multilabel': True,
    'frame_shift_ms': 10,
    'low_freq': 100,
    'high_freq': 11000
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models_vit.__dict__['vit_base_patch16']( 

    num_classes=32,
    drop_path_rate=0.1,
    global_pool=True,
    mask_2d=True,
    use_custom_patch=False
)

img_size=(128, 128)
in_chans= 1
emb_dim = 768 

model.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=in_chans, embed_dim=emb_dim, stride=16) # no overlap. stride=img_size=16
num_patches = model.patch_embed.num_patches
# num_patches = 64 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

model_path = model_pth_path

checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.to(device)

default_threshold = 1.1 

conn = psycopg2.connect(
    host="localhost",
    database="birdsong",
    user="postgres",
    password="Tfri23039978"
)

for audio_path in tqdm(glob.glob(os.path.join(audio_dir, '*.wav')), desc="Running Inference"):
    base_name = os.path.basename(audio_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    parts = file_name_without_extension.split('_')
    studio_name = parts[0]
    recording_date = parts[1]
    
    if convert_to_mono_switch:
        mono_audio_path = audio_mono_dir_setting / Path(audio_path).name
        convert_to_mono(audio_path, mono_audio_path)
        dataset = load_input_data(mono_audio_path, audio_conf)
    else:
        dataset = load_input_data(audio_path, audio_conf)

    input_data = dataset[1][0].unsqueeze(0)
    probabilities = inference(model, input_data, device)
    all_probabilities = None
    all_segments = None
    for i in range(len(dataset)):
        input_data, _ = dataset[i]
        input_data = input_data.unsqueeze(0).to(device)
        probabilities = inference(model, input_data, device)
        probabilities_np = probabilities.squeeze()
        probabilities_np = probabilities_np.reshape(1, -1)
        if all_probabilities is None:
            all_probabilities = probabilities_np
            all_segments = np.array([[dataset[i][1], dataset.segments[i]['start_time'], dataset.segments[i]['end_time']]])
        else:
            all_probabilities = np.concatenate((all_probabilities, probabilities_np), axis=0)
            all_segments = np.concatenate((all_segments, np.array([[dataset[i][1], dataset.segments[i]['start_time'], dataset.segments[i]['end_time']]])), axis=0)

    df = pd.DataFrame(all_probabilities)
    df.columns = ['Label: ' + str(i) for i in range(all_probabilities.shape[1])]
    df['audio_path'] = all_segments[:, 0]
    df['start_time'] = all_segments[:, 1]
    df['end_time'] = all_segments[:, 2]
    df.index = range(1, len(df) + 1)

    df['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df['sum'] = df[numeric_columns].sum(axis=1)

    df[['peak_label', 'peak_probability']] = df.apply(lambda row: get_peak_info(row, thresholds, default_threshold, all_probabilities.shape[1]), axis=1, result_type='expand')

    # Rename the label columns to use the display names
    label_columns = [f'Label: {i}' for i in range(all_probabilities.shape[1])]
    label_display_names = [label_mapping.get(i, f'Label: {i}') for i in range(all_probabilities.shape[1])]
    df = df.rename(columns=dict(zip(label_columns, label_display_names)))

    # Rearrange columns
    columns_order = ['created_at', 'audio_path', 'start_time', 'end_time'] + label_display_names + ['sum', 'peak_label', 'peak_probability']
    df = df[columns_order]

    output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{file_name_without_extension}.csv")

    try:
        df.to_csv(output_file_path, index_label="Index")
    except Exception as e:
        print(f"Error saving CSV file at {output_file_path}: {e}")
        sys.exit(1)

