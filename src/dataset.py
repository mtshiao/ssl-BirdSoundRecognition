# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# AST: https://github.com/YuanGongND/ast
# --------------------------------------------------------
import csv, os, sys
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
import torch.distributed as dist
import random
import math
import torch.nn.functional as F
import librosa

class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
            self, sampler, dataset,
            num_replicas=None,
            rank=None,
            shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle)
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            print(f"\n DistributedSamplerWrapper :  {indices[:10]} \n\n")
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
        
class DistributedWeightedSampler(Sampler):
    #dataset_train, samples_weight,  num_replicas=num_tasks, rank=global_rank
    def __init__(self, dataset, weights, num_replicas=None, rank=None, replacement=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.weights = torch.from_numpy(weights)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # targets = self.dataset.targets
        # # select only the wanted targets for this subsample
        # targets = torch.tensor(targets)[indices]
        # assert len(targets) == self.num_samples
        # # randomly sample this subset, producing balanced classes
        # weights = self.calculate_weights(targets)
        weights = self.weights[indices]

        subsample_balanced_indicies = torch.multinomial(weights, self.num_samples, self.replacement)
        # now map these target indicies back to the original dataset index...
        dataset_indices = torch.tensor(indices)[subsample_balanced_indicies]
        return iter(dataset_indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, use_fbank=False, fbank_dir=None, roll_mag_aug=False, load_video=False, mode='train'):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """

        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)
        self.use_fbank = use_fbank
        self.fbank_dir = fbank_dir

        self.data = data_json['data']
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        if 'multilabel' in self.audio_conf.keys():
            self.multilabel = self.audio_conf['multilabel']
        else:
            self.multilabel = False
        print(f'multilabel: {self.multilabel}')
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # print('Dataset: {}, mean {:.3f} and std {:.3f}'.format(self.dataset, self.norm_mean, self.norm_std))
        self.noise = self.audio_conf.get('noise')

        ######
        self.frame_shift_ms = self.audio_conf.get('frame_shift_ms')
        self.low_freq = self.audio_conf.get('low_freq')
        self.high_freq = self.audio_conf.get('high_freq')
        ######
        
        #########
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        #########

        if self.noise == True:
            print('now use noise augmentation')
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        self.roll_mag_aug=roll_mag_aug
        print(f'number of classes: {self.label_num}')
        print(f'size of dataset {self.__len__()}')

        self.cls_num = self._get_cls_num()  # Call to get the class occurrences

    # def compute_cls_num(self):
    #     cls_counter = np.zeros(self.label_num) # Initialize counters for each class
    #     for datum in self.data:
    #         labels = datum['labels'].split(',')
    #         for label_str in labels:
    #             cls_counter[int(self.index_dict[label_str])] += 1
    #     return cls_counter

    def _get_cls_num(self):
        cls_count = [0] * self.label_num
        for datum in self.data:
            for label_str in datum['labels'].split(','):
                cls_count[int(self.index_dict[label_str])] += 1
        return cls_count

    def _repeat_audio_segment(self, waveform, sr, start_time, end_time):
        start_time = start_time if start_time is not None else 0
        end_time = end_time if end_time is not None else waveform.shape[1] / sr

        duration = end_time - start_time
        if duration < 1.0:
            repetitions = int(np.ceil(1.0 / duration))
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = waveform[:, start_sample:end_sample]
            repeated_segment = torch.cat([segment] * repetitions, dim=1)
            return repeated_segment[:, :int(sr)]
        else:
            return waveform[:, int(start_time * sr):int(end_time * sr)]

    def _roll_mag_aug(self, waveform):
        waveform=waveform.numpy()
        idx=np.random.randint(len(waveform))
        rolled_waveform=np.roll(waveform,idx)
        mag = np.random.beta(10, 10) + 0.5
        return torch.Tensor(rolled_waveform*mag)

    def _wav2fbank(self, filename, filename2=None, start_time=None, end_time=None): #new
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            
            # new
            if start_time is not None or end_time is not None:
                # start_sample = int(start_time * sr) if start_time else 0
                # end_sample = int(end_time * sr) if end_time else None
                # waveform = waveform[:, start_sample:end_sample]
                waveform = self._repeat_audio_segment(waveform, sr, start_time, end_time)

            waveform = waveform - waveform.mean()
            if self.roll_mag_aug:
                waveform = self._roll_mag_aug(waveform)
 
 ## mixup augmentation
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            # new
            if start_time is not None or end_time is not None:
                # start_sample1 = int(start_time * sr) if start_time else 0
                # end_sample1 = int(end_time * sr) if end_time else None
                # waveform1 = waveform1[:, start_sample1:end_sample1]

                # start_sample2 = int(start_time * sr) if start_time else 0
                # end_sample2 = int(end_time * sr) if end_time else None
                # waveform2 = waveform2[:, start_sample2:end_sample2]
                waveform1 = self._repeat_audio_segment(waveform1, sr, start_time, end_time)
                waveform2 = self._repeat_audio_segment(waveform2, sr, start_time, end_time)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if self.roll_mag_aug:
                waveform1 = self._roll_mag_aug(waveform1)
                waveform2 = self._roll_mag_aug(waveform2)

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

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
            # print(' ######## use use_interpolation')
            if use_interpolation:
                # Reshape fbank for interpolation (add channel and batch dimension)
                fbank = fbank.unsqueeze(0).unsqueeze(0)
                # Interpolate
                fbank = F.interpolate(fbank, size=(target_length, self.melbins), mode='nearest') # nearest
                # fbank = F.interpolate(fbank, size=(target_length, fbank.shape[-1]), mode='bilinear') # bilinear
                # Remove added dimensions
                fbank = fbank.squeeze(0).squeeze(0)
            else:
                # Zero padding
                if p > 0:
                    m = torch.nn.ZeroPad2d((0, 0, 0, p))
                    fbank = m(fbank)
                elif p < 0:
                    fbank = fbank[:target_length, :]

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def _fbank(self, filename, filename2=None, start_time=None, end_time=None): #new
        if filename2 == None:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fbank = np.load(fn1)

            # new
            if start_time is not None or end_time is not None:
                # start_frame = int(start_time * 1000 / self.frame_shift_ms) if start_time else 0
                # end_frame = int(end_time * 1000 / self.frame_shift_ms) if end_time else None
                # fbank = fbank[start_frame:end_frame, :]
                waveform, sr = torchaudio.load(filename)
                # sr = 16000  # Assuming a sample rate of 16000 Hz, adjust this value if needed
                waveform = torch.from_numpy(fbank)
                waveform = self._repeat_audio_segment(waveform, sr, start_time, end_time)
                fbank = waveform.numpy()

            return torch.from_numpy(fbank), 0
        else:
            fn1 = os.path.join(self.fbank_dir, os.path.basename(filename).replace('.wav','.npy'))
            fn2 = os.path.join(self.fbank_dir, os.path.basename(filename2).replace('.wav','.npy'))
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            # new
            fbank1 = np.load(fn1)
            fbank2 = np.load(fn2)

            # new
            if start_time is not None or end_time is not None:
                start_frame1 = int(start_time * 1000 / self.frame_shift_ms)
                end_frame1 = int(end_time * 1000 / self.frame_shift_ms)
                fbank1 = fbank1[start_frame1:end_frame1, :]

                start_frame2 = int(start_time * 1000 / self.frame_shift_ms)
                end_frame2 = int(end_time * 1000 / self.frame_shift_ms)
                fbank2 = fbank2[start_frame2:end_frame2, :]

                duration = (end_frame1 - start_frame1) * self.frame_shift_ms / 1000
                if duration < 1.0:
                    repetitions = int(np.ceil(1.0 / duration))
                    segment1 = fbank1
                    segment2 = fbank2
                    repeated_segment1 = np.vstack([segment1] * repetitions)[:int(1000 / self.frame_shift_ms), :]
                    repeated_segment2 = np.vstack([segment2] * repetitions)[:int(1000 / self.frame_shift_ms), :]
                    fbank1 = repeated_segment1
                    fbank2 = repeated_segment2

            fbank = mix_lambda * np.load(fn1) + (1-mix_lambda) * np.load(fn2)  
            return torch.from_numpy(fbank), mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup: # for audio_exp, when using mixup, assume multilabel
            datum = self.data[index]
            
            # new
            start_time = datum.get('start_time', 0)
            end_time = datum.get('end_time', None)

            # find another sample to mix, also do balance sampling
            # sample the other sample from the multinomial distribution, will make the performance worse
            # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
            # sample the other sample from the uniform distribution
            mix_sample_idx = random.randint(0, len(self.data)-1)
            mix_datum = self.data[mix_sample_idx]

            # get the mixed fbank
            if not self.use_fbank:
                fbank, mix_lambda = self._wav2fbank(datum['wav'], mix_datum['wav'], start_time, end_time) #new
            else:
                fbank, mix_lambda = self._fbank(datum['wav'], mix_datum['wav'], start_time, end_time) #new
            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            datum = self.data[index]

            # new
            start_time = datum.get('start_time', 0)
            end_time = datum.get('end_time', None)

            label_indices = np.zeros(self.label_num)
            if not self.use_fbank:
                # fbank, mix_lambda = self._wav2fbank(datum['wav'], start_time, end_time) #new
                fbank, mix_lambda = self._wav2fbank(datum['wav'], None, start_time, end_time) #new
            else:
                fbank, mix_lambda = self._fbank(datum['wav'], None, start_time, end_time) #new
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            if self.multilabel:
                label_indices = torch.FloatTensor(label_indices)
            else:
                # remark : for ft cross-ent
                label_indices = int(self.index_dict[label_str])
        # SpecAug for training (not for eval)
        ## random masking augmentation
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = fbank.transpose(0,1).unsqueeze(0) # 1, 128, 1024 (...,freq,time)

        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank) # (..., freq, time)
        fbank = torch.transpose(fbank.squeeze(), 0, 1) # time, freq

        # fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        ###############
        # normalize the input for both training and test
        if not self.skip_norm: 
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass
        ###############

## noise and roll augmentation
        if self.noise == True: # default is false, true for spc
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank.unsqueeze(0), label_indices, datum['wav']

    def __len__(self):
        return len(self.data)

