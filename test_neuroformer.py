import wandb
import pandas as pd
import os
import torch
import sys

sys.path.append('/mnt/storage_6TB/s_calcagno/projects/neuroai/')
from src.datasets.allen_singlestimulus import STIM_RESPONSE_FRAMES, BASELINE_FRAMES, AllenSingleStimulusDataset, get_experiment_id
from utils import compute_classification_metrics, compute_metrics
import argparse
from neuroformer.model_neuroformer import load_model_and_tokenizer
from torchvision.transforms import Resize, CenterCrop

parser = argparse.ArgumentParser()
parser.add_argument('--stim', type=str)
parser.add_argument('--res', type=int, default=None)
parser.add_argument('--out-dir', '-o', type=str, default='.')
#parser.add_argument('--eval-on-acts', '-e', action='store_true')
parser.add_argument('--negative-only', '-n', action='store_true')

args = parser.parse_args()
print(args)
STIMULUS = args.stim
OUTPUT_DIR = args.out_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

neuroformer_df = pd.read_csv("neuroformer_forecasting.csv")

print(f"Testing on {STIMULUS}")
print("You are providing res: ", args.res)

if args.res is not None:
    RES = args.res
else:
    RES = STIM_RESPONSE_FRAMES[STIMULUS]
    
print(f"Response length: {RES}")
THRESHOLD = 0.1

stim_df = neuroformer_df[neuroformer_df['stimulus'] == STIMULUS]

nan_metrics = {
    "/mse": torch.nan,
    "/mae": torch.nan,
    "/mase_f0": torch.nan,
    "/mase_avg": torch.nan,
    "/smape": torch.nan,
    "/correlation": torch.nan,
    "/ssim": torch.nan,
    "/psnr": torch.nan,
    "/dtw": torch.nan,
}

def get_acts(tgt, res_len):
    # get the activations for the target
    return (tgt[:, :, :res_len].mean(dim=-1) > 0.1).int()

def get_only_pos(src, tgt, pred, acts):
    
    src_pos = src[acts==1].unsqueeze(0)
    tgt_pos = tgt[acts==1].unsqueeze(0)
    pred_pos = pred[acts==1].unsqueeze(0)
    
    return src_pos, tgt_pos, pred_pos

def get_only_neg(src, tgt, pred, acts):
    
    src_pos = src[acts==0].unsqueeze(0)
    tgt_pos = tgt[acts==0].unsqueeze(0)
    pred_pos = pred[acts==0].unsqueeze(0)
    
    return src_pos, tgt_pos, pred_pos


def norm_with_gradients(src, tgt, pred, res_len, eval_on_acts=False):

    src_norm = torch.zeros_like(src)
    tgt_norm = torch.zeros_like(tgt)
    pred_norm = torch.zeros_like(pred)
    
    
    for i, sample in enumerate(pred):
        for n, neuron in enumerate(sample):
            src_ = src[i, n]
            tgt_ = tgt[i, n]
            
            # apply smoothing on tgt
            tgt_tmp = tgt_.clone()
            for t in range(1, res_len):
                tgt_[t] = tgt_tmp[t-1:t+2].mean()
            
            gradient_src = torch.gradient(src_)[0]
            gradient_tgt = torch.gradient(tgt_[:res_len])[0]
            gradient_pred = torch.gradient(neuron[:res_len])[0]
            
            acc_grad_src = gradient_src.abs().sum()
            acc_grad_tgt = gradient_tgt.abs().sum()
            acc_pred_tgt = gradient_pred.abs().sum()
            
            #normalize
            src_ = src_ / acc_grad_src
            tgt_ = tgt_ / acc_grad_tgt
            neuron = neuron / acc_pred_tgt
            
            src_norm[i, n] = src_
            tgt_norm[i, n] = tgt_
            pred_norm[i, n] = neuron
                
    return src_norm, tgt_norm, pred_norm

def get_dataset(container_id, stimulus, split='test'):
    
    manifest = "/mnt/storage_6TB/s_calcagno/projects/datasets/allen"
    input_dir = "/mnt/storage_6TB/s_calcagno/projects/neuroai/data/allen"
    experiment_id = get_experiment_id(container_id, stimulus)

        
    dataset = AllenSingleStimulusDataset(
        manifest_file=manifest,
        split_dir=input_dir,
        split=split,
        experiment_id=experiment_id,
        stimulus=stimulus,
        source_length=96,
        forecast_length=64,
        stimuli_format ="raw",
        trace_format="corrected_fluorescence_dff",
        labels_format="tsai_wen",
        threshold=0.1,
        monitor_height=30,
    )
    
    stim = dataset.stimuli

    if stimulus == "natural_scenes":
        resize = Resize(79)
        center_crop = CenterCrop((30, 100))
        stim = center_crop(resize(torch.tensor(stim)))
    elif stimulus == "locally_sparse_noise":
        resize = Resize((30, 100))
        stim = resize(torch.tensor(stim))
    elif stimulus == "static_gratings":
        max_idx = max(stim.keys())
        stim_ = torch.zeros((max_idx + 1, 30, 100))
        center_crop = CenterCrop((30, 100))
        stim = {k: center_crop(torch.tensor(s)) for k, s in stim.items()}
        for k, s in stim.items():
            stim_[k] = s
        stim = stim_
    elif stimulus== "drifting_gratings":
        max_idx = max(stim.keys())
        stim_ = torch.zeros((max_idx + 1, 60, 30, 100))
        center_crop = CenterCrop((30, 100))
        stim = {k: center_crop(torch.tensor(s)) for k, s in stim.items()}
        for k, s in stim.items():
            stim_[k] = s
        stim = stim_
        # resample (every 5 frames)
        stim = stim[:, ::5]
        
    dataset.stimuli = stim
            
    return dataset

from torch.utils.data import DataLoader
from tqdm import tqdm

def all_device(data, device):
    device = torch.device(device)
    if isinstance(data, dict):
        return {k: all_device(v, device) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [all_device(v, device) for v in data if v is not None]
    elif isinstance(data, tuple):
        return tuple(all_device(v, device) for v in data if v is not None)
    else:
        return data.to(device)
    
def get_all_predictions(dataset, model):
    
    # Create dataloader
    if STIMULUS == "drifting_gratings":
        bs = 1
    else:
        bs = 16
    dataloader = DataLoader(dataset, batch_size=bs*5, shuffle=False, drop_last=False, num_workers=0)
    device='cuda'
        
    # move to device
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    # Get all predictions
    all_predictions = []
    all_targets = []
    all_src = []
    all_acts = []

    for batch in tqdm(dataloader):
        
        x = {
            'id_prev': batch['src'],
            'id': batch['src'][:, :, -1:],
            'frames': batch['stim'],
            'pad': None,
        }
        y = batch['tgt']
        activation_labels = batch['activation_labels']
        
        x = all_device(x, device)
        y = all_device(y, device)
                
        x['pad'] = None
        
        # init  output
        outputs = torch.zeros_like(y)
    
        # Forward
        for t in range(64):
            
            # Forward
            output, _, _ = model(x, None)
            
            # Check NaN
            if torch.isnan(output).any():
                raise FloatingPointError('Found NaN values')
            
            # Store output
            tgt_tmp = torch.cat([x['id'], output[:, :, -1:]], dim=-1)
            x['id'] = tgt_tmp
            
            outputs[:, :, t] = output[:, :, -1]
            
        pred = outputs
        
        # add to list
        all_predictions.append(pred.cpu().detach())
        all_targets.append(y.cpu().detach())
        all_src.append(x['id_prev'].cpu().detach())
        all_acts.append(activation_labels.cpu().detach())
        
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_src = torch.cat(all_src, dim=0)
    all_acts = torch.cat(all_acts, dim=0)
    
    return all_src, all_targets, all_predictions, all_acts

for i, exp in stim_df.iterrows():
        
    # load dataset
    #train_dataset = get_dataset(exp.container_id, exp.stimulus, "train", neuroformer=True)
    test_dataset = get_dataset(exp.container_id, exp.stimulus, "test")

    # init metrics
    metrics = {}

    base_path = ""
    try:
        config, tokenizer, model = load_model_and_tokenizer(os.path.join(base_path, exp.model_path))

        src, tgt, pred, acts = get_all_predictions(test_dataset, model)
        #if args.eval_on_acts:
        acts = get_acts(tgt, RES)
        if args.negative_only:
            src, tgt, pred = get_only_neg(src, tgt, pred, acts)
        else:
            src, tgt, pred = get_only_pos(src, tgt, pred, acts)
        
        src_norm, tgt_norm, pred_norm = norm_with_gradients(src, tgt, pred, res_len=RES)
        metrics["neuroformer"] = compute_metrics(src_norm, tgt_norm[:, :, :RES], pred_norm[:, :, :RES], tgt_norm[:, :, :RES], "")
    except FileNotFoundError:
        metrics["neuroformer"] = nan_metrics
    
    
    # save metrics
    metrics_df = pd.DataFrame(metrics)
    if args.res is not None:
        metrics_df.to_csv(os.path.join(OUTPUT_DIR, f"metrics_{exp.stimulus}_{exp.container_id}_res{args.res}.csv"))
    else:
        metrics_df.to_csv(os.path.join(OUTPUT_DIR, f"metrics_{exp.stimulus}_{exp.container_id}.csv"))