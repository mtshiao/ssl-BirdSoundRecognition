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

from pathlib import Path

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
    'nb_classes': 32, # Number of target class
    'input_size': 128,  
    'device': 'cuda', 
    'seed': 0, 
    'resume': '', 
    'start_epoch': 0, 
    'eval': False,  
    'dist_eval': False, 
    'first_eval_ep': 0, 
    'num_workers': 3, 
    'pin_mem': True, 
    'dataset': 'birdsong',
    'low_freq':100, #Hz
    'high_freq':11000, 


    'mean':-7.535187,
    'std':2.8788178,

    # augment
    'freqm': 36, # pixel
    'timem': 36, # pixel
    'roll_mag_aug': True, # False
    'mixup': 0.5,
    'noise': True, # 
    'cutmix': 0, # 
    'cutmix_minmax': None,
    'mixup_prob': 1.0,
    'mixup_switch_prob': 0.5,
    'mixup_mode': 'batch',
    'global_pool': True, # 
    'mask_2d': True,    # 
    'mask_t_prob': 0.2, # 
    'mask_f_prob': 0.2, # 
    ##

    'use_custom_patch': False,
    
    'epoch_len': 200000, 
    'weight_sampler': False, 
    'distributed_wrapper': True, # False
    'replacement': True, # False
    ##

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

data_dir = Path("/your/finetune/audio/data/path") 
output_dir = Path(__file__).resolve().parent / "output" 
model_dir = Path(__file__).resolve().parent / "model" 

data_train_relative = data_dir / "finetune_train.json"
data_val_relative = data_dir / "finetune_val.json"
data_test_relative = data_dir / "finetune_test.json"
label_csv_relative = data_dir / "finetune_labels_indices.csv"
output_dir_relative = model_dir 
pretrain_relative = model_dir / "model_01.pth"

# 
args['data_train'] = str(data_train_relative)
args['data_val'] = str(data_val_relative)
args['data_eval'] = str(data_test_relative)
args['label_csv'] = str(label_csv_relative)
args['output_dir'] = str(output_dir_relative)
args['log_dir'] = str(output_dir_relative)
args['pretrain'] = str(pretrain_relative)  

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


misc.init_distributed_mode(args)

# print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
cwd = os.getcwd()
print('Current directory:', cwd)
print("{}".format(args).replace(', ', ',\n'))

device = torch.device(args['device'])

seed = args['seed'] + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True


norm_stats = {'birdsong':[args['mean'], args['std']]}
target_length = {'birdsong':args['input_size']} # 
multilabel_dataset = {'birdsong': True}
audio_conf_train = {'num_mel_bins': 128, # 
                    'target_length': target_length[args['dataset']],
                    'freqm': args['freqm'],
                    'timem': args['timem'],
                    'mixup': args['mixup'],
                    'dataset': args['dataset'],
                    'mode': 'train',
                    'mean': args['mean'],
                    'std': args['std'],
                    'noise': args['noise'],
                    'multilabel': multilabel_dataset[args['dataset']], 
                    'frame_shift_ms': 10, 
                    'low_freq': args['low_freq'], 
                    'high_freq': args['high_freq']
                    }

audio_conf_val = {'num_mel_bins': 128, 
                    'target_length': target_length[args['dataset']], 
                    'freqm': 0,
                    'timem': 0,
                    'mixup': 0,
                    'dataset': args['dataset'],
                    'mode': 'val',
                    'mean': args['mean'],
                    'std': args['std'],
                    'noise': False,
                    'multilabel': multilabel_dataset[args['dataset']],
                    'frame_shift_ms': 10, # 7.687
                    'low_freq': args['low_freq'],
                    'high_freq': args['high_freq']
                    }

audio_conf_eval = {'num_mel_bins': 128, 
                    'target_length': target_length[args['dataset']], 
                    'freqm': 0,
                    'timem': 0,
                    'mixup': 0,
                    'dataset': args['dataset'],
                    'mode': 'eval',
                    'mean': args['mean'],
                    'std': args['std'],
                    'noise': False,
                    'multilabel': multilabel_dataset[args['dataset']],
                    'frame_shift_ms': 10, # 7.687
                    'low_freq': args['low_freq'],
                    'high_freq': args['high_freq']
                    }

dataset_train = AudiosetDataset(
                    args['data_train'],
                    label_csv = args['label_csv'],
                    audio_conf = audio_conf_train, 
                    use_fbank = args['use_fbank'],
                    fbank_dir = args['fbank_dir'], 
                    roll_mag_aug = args['roll_mag_aug'],
                    load_video = args['load_video'],
                    mode = 'train'
                )

dataset_val = AudiosetDataset(
                args['data_val'],
                label_csv = args['label_csv'],
                audio_conf = audio_conf_val, 
                use_fbank = args['use_fbank'],
                fbank_dir = args['fbank_dir'], 
                roll_mag_aug = args['roll_mag_aug'],
                load_video = args['load_video'],
                mode = 'val'
            )

dataset_eval = AudiosetDataset(args['data_eval'],
                label_csv = args['label_csv'],
                audio_conf = audio_conf_eval,
                use_fbank = args['use_fbank'],
                fbank_dir = args['fbank_dir'],
                roll_mag_aug = False,
                load_video = args['load_video'],
                mode = 'eval'
            )

if True: #args.distributed:
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    num_nodes = int(os.environ.get('num_nodes', 1))
    ddp = int(os.environ.get('DDP', 1))
    num_nodes = max(ddp, num_nodes)
    rank = int(os.environ.get('NODE_RANK', 0))
    print(f"num_nodes:{num_nodes}, rank:{rank}, ddp:{ddp}, num_tasks:{num_tasks}, global_rank:{global_rank}")
    # num_nodes:1, rank:0, ddp:1, num_tasks:8, global_rank:0 (sbatch)

    if args['weight_sampler']:
        samples_weight = np.loadtxt(args['weight_csv'], delimiter=',')
        if args['distributed_wrapper']:
            print('use distributed_wrapper sampler')
            epoch_len=args['epoch_len'] #200000 #=> 250000
            #epoch_len=21000 # AS-20K
            # replacement should be False
            sampler_train = DistributedSamplerWrapper(
                                sampler = WeightedRandomSampler(samples_weight, num_samples = epoch_len, replacement = args['replacement']),
                                dataset = range(epoch_len),
                                num_replicas = num_tasks, #num_nodes, #num_tasks?
                                rank = global_rank, #rank, # global_rank?
                            )
            
            val_epoch_len = len(dataset_val)
            sampler_val = torch.utils.data.DistributedSampler(
                                dataset_val, 
                                num_replicas = num_tasks,
                                rank = global_rank,
                                shuffle = False,
                                replacement = args['replacement']
                            )

        else:
            #sampler_train = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
            sampler_train = DistributedWeightedSampler(
                                dataset_train,
                                samples_weight,
                                num_replicas = num_tasks,
                                rank = global_rank,
                                replacement = args['replacement']
                            )
            
            sampler_val = torch.utils.data.DistributedSampler(
                                dataset_val, 
                                num_replicas = num_tasks,
                                rank = global_rank,
                                shuffle = False,
                                replacement = args['replacement']
                            )
    
    else:
        sampler_train = torch.utils.data.DistributedSampler(
                            dataset_train,
                            num_replicas = num_tasks,
                            rank = global_rank,
                            shuffle = True
                        )
        
        sampler_val = torch.utils.data.DistributedSampler(
                            dataset_val,
                            num_replicas = num_tasks,
                            rank = global_rank,
                            shuffle = False
                        )

    print("Sampler_train = %s" % str(sampler_train))
    if args['dist_eval']:
        if len(dataset_eval) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            
        sampler_eval = torch.utils.data.DistributedSampler(
                            dataset_eval,
                            num_replicas = num_tasks,
                            rank = global_rank,
                            shuffle = True   # shuffle = True to reduce monitor bias
                        )
    else:
        sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)

else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)


if global_rank == 0 and args['log_dir'] is not None and not args['eval']:
    os.makedirs(args['log_dir'], exist_ok=True) 
    log_writer = SummaryWriter(log_dir=args['log_dir'])
else:
    log_writer = None


# PyTorch DataLoader
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, 
    sampler = sampler_train, 
    batch_size = args['batch_size'], 
    num_workers = args['num_workers'], 
    pin_memory = args['pin_mem'],
    drop_last = True, 
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    sampler = sampler_val,
    batch_size = args['batch_size'],
    num_workers = args['num_workers'],
    pin_memory = args['pin_mem'],
    drop_last = False,
)

data_loader_eval = torch.utils.data.DataLoader(
    dataset_eval,
    sampler = sampler_eval,
    batch_size = args['batch_size'],
    num_workers = args['num_workers'],
    pin_memory = args['pin_mem'],
    drop_last = False
)

mixup_fn = None
mixup_active = args['mixup'] > 0 or args['cutmix'] > 0. or args['cutmix_minmax'] is not None
if mixup_active:
    print("Mixup is activated!")
    mixup_fn = Mixup(
        mixup_alpha = args['mixup'],
        cutmix_alpha = args['cutmix'],
        cutmix_minmax = args['cutmix_minmax'],
        prob = args['mixup_prob'],
        switch_prob = args['mixup_switch_prob'],
        mode = args['mixup_mode'],
        label_smoothing = args['smoothing'],
        num_classes = args['nb_classes']
    )


model = models_vit.__dict__[args['model']](
    num_classes = args['nb_classes'], 
    drop_path_rate = args['drop_path'], 
    global_pool = args['global_pool'], 
    mask_2d = args['mask_2d'], 
    use_custom_patch = args['use_custom_patch'],

)

if args['audio_exp']:
    img_size=(target_length[args['dataset']],128) # img_size （128, 128）
    in_chans=1 # 
    emb_dim = 768 # embedding dimension
    if args['model'] == "vit_large_patch16":
        emb_dim = 1024
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
        
        model.pos_embed = nn.Parameter(torch.zeros(1, 12 * 12 + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

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

# model.pos_embed = nn.Parameter(torch.zeros(1, 64 + 1, 768), requires_grad=False)  # fixed sin-cos embedding


# import pre-trained model:
if args['pretrain']: 
    checkpoint = torch.load(args['pretrain'], map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args['pretrain'])
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()

    if not args['eval']:
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint") 
                del checkpoint_model[k]
    # load pre-trained model

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # manually initialize fc layer
    if not args['eval']:
        trunc_normal_(model.head.weight, std=2e-5)


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


# 
if args['distributed']:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']])
    model_without_ddp = model.module

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

# 
print(f"Start training for {args['epochs']} epochs")
start_time = time.time()
max_mAP = 0.0


for epoch in range(args['start_epoch'], args['epochs']):
    if args['distributed']: 
        data_loader_train.sampler.set_epoch(epoch)
        data_loader_val.sampler.set_epoch(epoch) 
    
    train_stats = train_one_epoch(
            model, criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args['clip_grad'],
            mixup_fn,
            log_writer = log_writer,
            args = args
        )
    
    val_stats = val_one_epoch(
            model,
            criterion,
            data_loader_val,
            device,
            log_writer = log_writer,
            args = args
        )

    if args['output_dir']:
        misc.save_model(
            args = args,
            model = model,
            model_without_ddp = model_without_ddp,
            optimizer = optimizer,
            loss_scaler = loss_scaler,
            epoch = epoch
        )

    if epoch >= args['first_eval_ep']:
        test_stats = evaluate(
            data_loader_eval,
            model,
            device,
            args['dist_eval']
        )

        print(f"mAP of the network on the {len(dataset_eval)} test images: {test_stats['mAP']:.4f}")
        max_mAP = max(max_mAP, test_stats["mAP"])
        print(f'Max mAP: {max_mAP:.4f}')
        
    else:
        test_stats = {'mAP': 0.0}
        print(f'too new to evaluate!')

    if log_writer is not None:  
        log_writer.add_scalar('perf/mAP', test_stats['mAP'], epoch)

    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                 **{f'val_{k}': v for k, v in val_stats.items()},
                 **{f'test_{k}': v for k, v in test_stats.items()},
                 'epoch': epoch,
                 'n_parameters': n_parameters
                 }

    if args['output_dir'] and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args['output_dir'], "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
    
    print() 

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))

