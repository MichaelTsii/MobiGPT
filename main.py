import argparse
import random
import os
from models_with_mask_scale_multiprompt_fix import DiT_models
from train import TrainLoop
import setproctitle
import torch
from DataLoader import data_load_main
from utils import *
import torch as th
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import datetime
import pickle

# 数据集基本参数
dataset_list = 'newtraffic-nj' # 'traffic_RSRP-CPGMCM-12Features-calc_sh-app-rebuild'
time_length = 64

task = 'long_prediction' # 'mix', 'short_prediction', 'long_prediction', 'generation'
prompt_state = 'few-shot' #'load', 'train', 'test', 'zero-shot', 'few-shot'
save_folder = 'traffic_RSRP-CPGMCM-12Features-calc_sh-app-rebuild__pretrained' # 仅当 prompt_state == 'test' 时有效
fewshot_rate = 0.1

use_cond = True
os.environ["CUDA_VISIBLE_DEVICES"] = "6" # cuda
process_name = dataset_list + "@qxq"

def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def dev(device_id='0'):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    return th.device("cpu")

def create_argparser():
    defaults = dict(
        data_dir="",
        lr=1e-4,
        task = 'short',
        early_stop = 20,
        weight_decay=1e-4,
        log_interval=20,

        # 基本参数 ——————————————
        batch_size = 32,
        total_epoches = 600,

        device_id='0',
        machine = 'machine_name',
        mask_ratio = 0.5,
        lr_anneal_steps = 300,
        patch_size = 1,
        random=True,
        t_patch_size = 1,
        size = 'small',
        clip_grad = 0.5,
        mask_strategy = ['generation_masking', 'random_masking', 'short_long_temporal_masking'], # 'random'
        mask_strategy_random = 'batch', # ['none','batch']
        # mask_strategy = 'generation_masking',
        # mask_strategy_random = 'none', # ['none','batch']
        use_cond = use_cond,
        mode='training',
        file_load_path = '',
        min_lr = 1e-5,
        dataset = dataset_list,
        stage = 0,
        no_qkv_bias = 0,
        pos_emb = 'SinCos',
        used_data = '',
        process_name = process_name,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    th.autograd.set_detect_anomaly(True)

    args = create_argparser().parse_args()
    setproctitle.setproctitle("{}-{}".format(args.dataset, args.total_epoches))
    setup_init(100)

    args.task = task
    args.prompt_state = prompt_state
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    args.folder = '{}/'.format(dataset_list + '_' + current_time)
    args.datatype = dataset_list
    args.time_length = time_length
    args.model_path = './experiments/{}'.format(args.folder) 
    logdir = "./logs/{}".format(args.folder)
    args.save_folder = save_folder
    args.fewshot_rate = fewshot_rate

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')
        os.mkdir(args.model_path+'scalers/')

    print('start data load')
    data, val_data, test_data, args.scaler = data_load_main(args)

    if args.prompt_state in ['train', 'zero-shot', 'few-shot']: # 保存scaler和
        for datatype in args.dataset.split('_'):
            with open(args.model_path + 'scalers/scaler_' + datatype + '.pk', "wb") as f:
                pickle.dump(args.scaler[datatype], f)

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    device = dev(args.device_id)

    # model = mae_vit_ST(args=args).to(device)
    model = DiT_models['DiT-S/8'](
        args=args,
        input_size=32,
        num_classes=1000,
    ).to(device)
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=1000)

    total_params = sum(p.numel() for p in model.parameters())
    print("模型参数总量:", total_params)

    # 如果直接读取已训练模型
    if args.prompt_state in ['train', 'load']:
        TrainLoop(
            args = args,
            writer = writer,
            model=model,
            diffusion = diffusion,
            data=data,
            test_data=test_data, 
            val_data=val_data,
            device=device
        ).run_loop(args)
    
    elif args.prompt_state in ['test', 'zero-shot']:
        model_folder = './experiments/' + save_folder + '/model_save/'
        file_path = model_folder + 'model_best_' + save_folder[:-12] + '.pkl'
        state_dict = torch.load(file_path)
        model.load_state_dict(state_dict)

        TrainLoop(
            args = args,
            writer = writer,
            model=model,
            diffusion = diffusion,
            data=data,
            test_data=test_data, 
            val_data=val_data,
            device=device
        ).evaluating()
    
    elif args.prompt_state in ['few-shot']:
        args.total_epoches = 300

        TrainLoop(
            args = args,
            writer = writer,
            model=model,
            diffusion = diffusion,
            data=data,
            test_data=test_data, 
            val_data=val_data,
            device=device
        ).run_loop(args)

if __name__ == "__main__":
    main()