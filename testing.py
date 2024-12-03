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
assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import to_2tuple

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed, interpolate_pos_embed_audio, interpolate_patch_embed_audio, interpolate_pos_embed_img2audio
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from src import models_vit

from src.engine_finetune import train_one_epoch, evaluate, val_one_epoch  #, train_one_epoch_av, evaluate_av
from src.dataset import AudiosetDataset, DistributedWeightedSampler, DistributedSamplerWrapper
from timm.models.vision_transformer import PatchEmbed

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Sampler

import csv
import math
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score, roc_auc_score, fbeta_score, precision_score, recall_score
from concurrent.futures import ProcessPoolExecutor

current_working_directory = Path.cwd()
script_dir = current_working_directory.joinpath('code')

args = {
    'batch_size': 256, # 4
    'epochs': 100, # 60
    'accum_iter': 1,
    'model': 'vit_base_patch16',
    'drop_path': 0.1,
    'clip_grad': None,
    'weight_decay': 0.0005,
    'lr': None,
    'blr': 1e-3, # 0.002
    'layer_decay': 0.75,
    'min_lr': 0.000001, 
    'warmup_epochs': 4,
    'smoothing': 0.1,
    'nb_classes': 32, 
    'input_size': 128,  
    'device': 'cuda', 
    'seed': 0, 
    'resume': '', 
    'start_epoch': 0, 
    'eval': True,  
    'dist_eval': False, 
    'first_eval_ep': 0, 
    'num_workers': 3, 
    'pin_mem': True,  
    'dataset': 'birdsong',

    'use_custom_patch': False,

    # augment
    'freqm': 0,
    'timem': 0,
    'roll_mag_aug': True, 
    'mixup': 0,
    'noise': True,
    'cutmix': 0, 
    'cutmix_minmax': None,
    'mixup_prob': 1.0,
    'mixup_switch_prob': 0.5,
    'mixup_mode': 'batch',
    'global_pool': True,
    'low_freq':100,
    'high_freq':11000,
    'mask_2d': True,    
    'mask_t_prob': 0.2, 
    'mask_f_prob': 0.2, 

    'mean':-7.535187,
    'std':2.8788178,
    
    'epoch_len': 200000, 
    'weight_sampler': False, 
    'distributed_wrapper': True, 
    'replacement': True, 

    'world_size': 2, 
    'local_rank': -1,
    'dist_on_itp': False, 
    'dist_url': 'env://',
    'distributed': True,
    'dist_backend': 'nccl',
    'rank': 0,
    'gpu': 0,
    #

    'audio_exp': True,
    'use_soft': False, 

    ## 
    'data_path': '/datasets01/imagenet_full_size/061417/', 
    'fbank_dir': "/checkpoint/berniehuang/ast/egs/esc50/data/ESC-50-master/fbank",
    'use_fbank': False,
    'load_video': False,
    'n_frm': 6,

    'load_imgnet_pt': False, 
    'source_custom_patch': False,    

    'aa': 'rand-m9-mstd0.5-inc1',
    'resplit': False,
    'replace_with_mae': False,

}

script_dir = Path(__file__).resolve().parent

#data_dir = Path("/your/test/audio/data/path")
data_dir = Path("/home/mtshi/Linux/Audio_data")
model_dir = Path(__file__).resolve().parent / "model" 

data_test_relative = data_dir / "finetune_test.json"
label_csv_relative = data_dir / "finetune_labels_indices.csv"
finetune_checkpoint_relative = model_dir / "finetune_model" / "model_01_1_1_1.pth"

args['data_eval'] = str(data_test_relative)
args['label_csv'] = str(label_csv_relative)
args['finetune'] = str(finetune_checkpoint_relative)  

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
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

cwd = os.getcwd()
print('Current directory:', cwd)
print("{}".format(args).replace(', ', ',\n'))

device = torch.device(args['device'])

seed = args['seed'] + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True

audio_conf_eval = {'num_mel_bins': 128, 
                    'target_length': args['input_size'], 
                    'freqm': args['freqm'],
                    'timem': args['timem'],
                    'mixup': args['mixup'],
                    'dataset': args['dataset'],
                    'mode': 'eval',
                    'mean': args['mean'],
                    'std': args['std'],
                    'noise': ['noise'],
                    'multilabel': True,
                    'frame_shift_ms': 10,
                    'low_freq': args['low_freq'],
                    'high_freq': args['high_freq']
                    }
dataset_eval = AudiosetDataset(args['data_eval'],
                label_csv = args['label_csv'],
                audio_conf = audio_conf_eval,
                use_fbank = args['use_fbank'],
                fbank_dir = args['fbank_dir'],
                roll_mag_aug = ['roll_mag_aug'],
                load_video = args['load_video'],
                mode = 'eval'
            )

sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

data_loader_eval = torch.utils.data.DataLoader(
    dataset_eval,
    sampler = sampler_eval,
    batch_size = args['batch_size'],
    num_workers = args['num_workers'],
    pin_memory = args['pin_mem'],
    drop_last = False
)

model = models_vit.__dict__[args['model']](
    num_classes = args['nb_classes'], 
    drop_path_rate = args['drop_path'], 
    global_pool = args['global_pool'], 
    mask_2d = args['mask_2d'], 
    use_custom_patch = args['use_custom_patch'], 
)

if args['audio_exp']:
    img_size=(128,128) 
    in_chans=1 
    emb_dim = 768 
    if args['model'] == "vit_small_patch16":
        emb_dim = 384
    if args['use_custom_patch']:
        model.patch_embed = PatchEmbed_new(
                                img_size = img_size,
                                patch_size = 16,
                                in_chans = 1,
                                embed_dim = emb_dim,
                                stride = 10 
        )
        model.pos_embed = nn.Parameter(torch.zeros(1, 1212 + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

    else:
        model.patch_embed = PatchEmbed_new(
                                img_size = img_size,
                                patch_size = (16,16),
                                in_chans = 1,
                                embed_dim = emb_dim,
                                stride = 16  # no overlap. stride=img_size=16
                            )
        
        num_patches = model.patch_embed.num_patches
        #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

#import finetune model
if args['finetune']: 
    checkpoint = torch.load(args['finetune'], map_location='cpu')
    print("Load finetune checkpoint from: %s" % args['finetune'])
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

model.to(device)

model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model = %s" % str(model_without_ddp))
print('number of params (M): %.2f' % (n_parameters / 1.e6))

eff_batch_size = args['batch_size'] * args['accum_iter'] * misc.get_world_size()

if args['lr'] is None:  # only base_lr is specified
    args['lr'] = args['blr'] * eff_batch_size / 256

print("base lr: %.2e" % (args['lr'] * 256 / eff_batch_size))
print("actual lr: %.2e" % args['lr'])

print("accumulate grad iterations: %d" % args['accum_iter'])
print("effective batch size: %d" % eff_batch_size)

# build optimizer with layer-wise lr decay (lrd)
param_groups = lrd.param_groups_lrd(
    model_without_ddp,
    args['weight_decay'],
    no_weight_decay_list = model_without_ddp.no_weight_decay(),
    layer_decay = args['layer_decay']
)

optimizer = torch.optim.AdamW(param_groups, lr=args['lr'])
loss_scaler = NativeScaler()

if args['use_soft']:
    criterion = SoftTargetCrossEntropy() 
else:
    criterion = nn.BCEWithLogitsLoss() # works better

print("criterion = %s" % str(criterion))

misc.load_model(
    args = args,
    model_without_ddp = model_without_ddp,
    optimizer = optimizer,
    loss_scaler = loss_scaler
)

if args['eval']:
    test_stats = evaluate(       
            data_loader_eval,
            model,
            device,
            args['dist_eval']
        )

    with open('aps.txt', 'w') as fp:
        aps = test_stats['AP']
        aps = [str(ap) for ap in aps]
        fp.write('\n'.join(aps))
    print(f"Accuracy of the network on the {len(dataset_eval)} test images: {test_stats['mAP']:.4f}")
    exit(0)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def to_prob(pre_file, apply_sigmoid=False):

    # Read pre.csv
    with open(pre_file, 'r') as file:
        pre_reader = csv.reader(file, delimiter=',')
        pre_data = list(pre_reader)

    pre_data = [[float(val) for val in row] for row in pre_data]
    if apply_sigmoid:
        pre_data = [[sigmoid(val) for val in row] for row in pre_data]

    with open(Path.cwd().parents[0].joinpath('report', 'model_new', 'testing', 'pre.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(pre_data)

to_prob(Path.cwd().parents[0].joinpath('report', 'model_new', 'testing', 'pre.csv'),
             apply_sigmoid=True)


def calculate_f_scores_with_optimal_thresholds(y_true, y_pred, beta, thresholds):
    f_scores = [fbeta_score(y_true[:, i], y_pred[:, i] > thresholds[i], beta=beta) for i in range(y_true.shape[1])]
    return f_scores

def _process_column(args):
    y_true_column, y_pred_column, beta = args
    unique_thresholds = np.unique(y_pred_column)
    best_score = -1
    best_threshold = None
    for threshold in unique_thresholds:
        score = fbeta_score(y_true_column, y_pred_column > threshold, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold if best_threshold is not None else 1.0

def find_optimal_thresholds(y_true, y_pred, beta):
    columns = [(y_true[:, i], y_pred[:, i], beta) for i in range(y_true.shape[1])]
    with ProcessPoolExecutor() as executor:
        optimal_thresholds = list(executor.map(_process_column, columns))

    print(optimal_thresholds)
    return optimal_thresholds

def calculate_precision_recall_scores(y_true, y_pred, thresholds):
    precision_scores = []
    recall_scores = []
    for i in range(y_true.shape[1]):
        binarized_predictions = y_pred[:, i] > thresholds[i]
        precision = precision_score(y_true[:, i], binarized_predictions)
        recall = recall_score(y_true[:, i], binarized_predictions)
        precision_scores.append(precision)
        recall_scores.append(recall)
    return precision_scores, recall_scores

def write_results_to_file(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")

def read_csv(file_path):
    return np.genfromtxt(file_path, delimiter=',')

def process_files(true_labels_file, predicted_probs_file, output_dir):
    true_labels = read_csv(true_labels_file)
    predicted_probs = read_csv(predicted_probs_file)

    lrap = label_ranking_average_precision_score(true_labels, predicted_probs)
    map_score = np.mean([average_precision_score(true_labels[:, i], predicted_probs[:, i]) for i in range(true_labels.shape[1])])
    ap_scores = [average_precision_score(true_labels[:, i], predicted_probs[:, i]) for i in range(true_labels.shape[1])]

    # Find optimal thresholds for F1 and F0.5
    optimal_f1_thresholds = find_optimal_thresholds(true_labels, predicted_probs, beta=1)
    optimal_f05_thresholds = find_optimal_thresholds(true_labels, predicted_probs, beta=0.5)

    # Calculate F1 and F0.5 scores for all classes using the optimal thresholds
    f1_scores = calculate_f_scores_with_optimal_thresholds(true_labels, predicted_probs, beta=1, thresholds=optimal_f1_thresholds)
    f05_scores = calculate_f_scores_with_optimal_thresholds(true_labels, predicted_probs, beta=0.5, thresholds=optimal_f05_thresholds)

    # Calculate Precision and Recall for F1 and F0.5 thresholds
    f1_precision_scores, f1_recall_scores = calculate_precision_recall_scores(true_labels, predicted_probs, optimal_f1_thresholds)
    f05_precision_scores, f05_recall_scores = calculate_precision_recall_scores(true_labels, predicted_probs, optimal_f05_thresholds)

    auc_roc_scores = [roc_auc_score(true_labels[:, i], predicted_probs[:, i]) for i in range(true_labels.shape[1])]

    with open(output_dir / 'scores.csv', 'w') as file:
        file.write("AP,ROC,F1,F1_precision,F1_recall,F05,F05_precision,F05_recall\n")
        for i in range(len(ap_scores)):
            file.write(f"{ap_scores[i]},{auc_roc_scores[i]},{f1_scores[i]},{f1_precision_scores[i]},{f1_recall_scores[i]},{f05_scores[i]},{f05_precision_scores[i]},{f05_recall_scores[i]}\n")

    print(f"LRAP: {lrap}")
    print(f"mAP: {map_score}")
    print(f"Scores are saved to {output_dir / 'scores.csv'}")

    # Write results to text files with specified path
    write_results_to_file(output_dir / "f1_scores.txt", f1_scores)
    write_results_to_file(output_dir / "f05_scores.txt", f05_scores)
    write_results_to_file(output_dir / "optimal_f1_thresholds.txt", optimal_f1_thresholds)
    write_results_to_file(output_dir / "optimal_f05_thresholds.txt", optimal_f05_thresholds)

    with open(output_dir / "lrap.txt", 'w') as file:
        file.write(f"{lrap}\n")

# Define relative paths based on project structure
project_root = Path(__file__).resolve().parent  
report_dir = project_root / "output" / "testing"

true_labels_file = report_dir / "tar.csv"
predicted_probs_file = report_dir / "pre.csv"
output_dir = report_dir

output_dir.mkdir(parents=True, exist_ok=True)

process_files(true_labels_file, predicted_probs_file, output_dir)

