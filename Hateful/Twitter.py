import argparse
import os
import ruamel.yaml as yaml
#import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import sys
sys.path.append("../")
from Hateful_graph_final.model_twitter import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from module.functions import CMD, DiffLoss

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from sklearn.metrics import f1_score,recall_score,precision_score, accuracy_score
import metrics
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

seed = 42 + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True


def get_cmd_loss(utt_shared_t, utt_shared_v):
    # losses between shared states
    loss_cmd = CMD()
    loss = loss_cmd(utt_shared_t, utt_shared_v, 5)
    # loss_coral = CorrelationAlignmentLoss()
    # loss = loss_coral(utt_shared_t, utt_shared_v)

    return loss

def get_diff_loss(utt_shared_t, utt_shared_v, utt_private_t, utt_private_v):
    ###1e-5
    loss_diff = DiffLoss()
    shared_t = utt_shared_t
    shared_v = utt_shared_v
    private_t = utt_private_t
    private_v = utt_private_v

    # Between private and shared
    loss = loss_diff(private_t, shared_t)
    loss += loss_diff(private_v, shared_v)
    # Across privates
    loss += loss_diff(private_t, private_v)

    return loss


def train(model, data_loader_hateful, optimizer, tokenizer, warmup_steps, device, scheduler, config, start_time, model_without_ddp, val_seen_loader_hateful, test_seen_loader_hateful, args):
        
    lr_scheduler = scheduler
    # train
    model.train()  
    len_data_loader_hateful = len(data_loader_hateful) - 1
    
    best_seen = 0
    best_seen_epoch = 0
    
    best_seen_f1 = 0
    best_seen_f1_epoch = 0

    best_seen_rec = 0
    best_seen_rec_epoch = 0

    best_seen_pre = 0
    best_seen_pre_epoch = 0

    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
  
    step_size = 100
    star_time=time.time()
    for i in range(0, config['schedular']['epochs']):


        if i % len_data_loader_hateful == 0:
            data_hateful = iter(data_loader_hateful)
        hateful_images, hateful_text, hateful_targets = next(data_hateful)

        hateful_images, hateful_targets = hateful_images.to(device,non_blocking=True), hateful_targets.to(device,non_blocking=True)

        text_inputs = tokenizer(hateful_text, padding='longest', return_tensors="pt").to(device) 

        if (i/1000)>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/(config['batch_size_train'] * 2))

        ly, utt_private_t, utt_private_v, utt_shared_t, utt_shared_v, kl_loss = model(hateful_images, text_inputs, targets=hateful_targets, train=True, alpha=alpha)  
        cmd_loss = get_cmd_loss(utt_shared_t, utt_shared_v)
        diff_loss = get_diff_loss(utt_shared_t, utt_shared_v, utt_private_t, utt_private_v)
        
        loss = ly + cmd_loss +  diff_loss  +  kl_loss

        optimizer.zero_grad()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()    
        lr=optimizer.param_groups[0]["lr"]       
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss.item())
        if i % 100 == 0:
            # runing_time=(time.time()-star_time)/3600
            # remain_time = (runing_time/(i+1))* config['schedular']['epochs']
            print('i={} lr={:.6f} loss={:.5f} ly={:.3f} cmd_loss={:.4f} diff_loss={:.4f}'.format(i, lr, loss, ly, cmd_loss, diff_loss))
        if i<=100 and i%step_size==0 and i<=config['schedular']['epochs']: 
            scheduler.step(i//step_size)      

        
        # gather the stats from all processes
        if i % 500 == 0:
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger.global_avg())     
            train_stats = {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    
            val_seen_stats = evaluate(model, val_seen_loader_hateful, tokenizer, device, config)
            test_seen_stats = evaluate(model, test_seen_loader_hateful, tokenizer, device, config)

            if utils.is_main_process():  
                if args.evaluate:
                    log_stats = {**{f'val_seen_{k}': v for k, v in val_seen_stats.items()},
                                **{f'test_seen_{k}': v for k, v in test_seen_stats.items()},
                                'step': i,
                                }

                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")                
                else:    
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_seen_{k}': v for k, v in val_seen_stats.items()},
                                **{f'test_seen_{k}': v for k, v in test_seen_stats.items()},
                                'step': i,
                                }

                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    if float(val_seen_stats['acc'])>best_seen:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': i,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_seen_best_acc.pth')) 
                        best_seen = float(val_seen_stats['acc'])
                        best_seen_epoch=i

                    if float(val_seen_stats['f1'])>best_seen_f1:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': i,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_seen_f1.pth')) 
                        best_seen_f1 = float(val_seen_stats['f1'])
                        best_seen_f1_epoch = i

                    if float(val_seen_stats['rec'])>best_seen_rec:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': i,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_seen_rec.pth')) 
                        best_seen_rec = float(val_seen_stats['rec'])
                        best_seen_rec_epoch = i

                    if float(val_seen_stats['pre'])>best_seen_pre:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': i,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_seen_pre.pth')) 
                        best_seen_pre = float(val_seen_stats['pre'])
                        best_seen_pre_epoch = i

            if args.evaluate:
                break
            epoch = int(i / 500 / 2)
            lr_scheduler.step(epoch+warmup_steps+1)  
            dist.barrier()

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str)) 
            model.train()
                
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best val seen acc step: %d"%best_seen_epoch)     
            f.write("best val seen f1 step: %d"%best_seen_f1_epoch)
            f.write("best val seen rec step: %d"%best_seen_rec_epoch)
            f.write("best val seen pre step: %d"%best_seen_pre_epoch)

@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    pre_list=[]
    label_list=[]

    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)   
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  

        prediction = model(images, text_inputs, targets=targets, train=False) 
        
        _, pred_class = prediction.max(1)
        #高版本pytorch的tensor和int之间的除法不能直接用'/'
        accuracy = torch.true_divide((targets==pred_class).sum(), targets.size(0))  

        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))
    
        pre_list.append(prediction)
        label_list.append(targets)

    pre_list = torch.cat(pre_list, 0)
    label_list = torch.cat(label_list, 0)
    
    _, pre_list = pre_list.max(1)
    label_list = label_list.unsqueeze(1) 
    # sklearn中的auc_roc计算
    # pre_list = F.softmax(pre_list, dim=-1)[:, 1]#[size, 1]
    # label_list = label_list.unsqueeze(1) 
    # auc_roc = roc_auc_score(label_list.cpu().numpy(), pre_list.cpu().numpy())

    f1 = f1_score(label_list.cpu().numpy(), pre_list.cpu().numpy())
    rec = recall_score(label_list.cpu().numpy(), pre_list.cpu().numpy())
    pre = precision_score(label_list.cpu().numpy(), pre_list.cpu().numpy())
    f1 = f1.item()
    rec = rec.item()
    pre = pre.item()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    print("Averaged stats: f1: ", f1)  
    print("Averaged stats: rec: ", rec)  
    print("Averaged stats: pre: ", pre)  
    score_dic = {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    score_dic['f1'] = f1
    score_dic['rec'] = rec
    score_dic['pre'] = pre
    return score_dic
    
    
def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility

    #### Dataset #### 
    print("Creating dataset")
    #需要更改
    datasets = create_dataset('twitter', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader_hateful, val_seen_loader_hateful, test_seen_loader_hateful = create_loader(
                                                          datasets,samplers,
                                                          batch_size=[config['batch_size_train']]*1+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],is_trains=[True,False,False], 
                                                          collate_fns=[None,None,None]
                                                        )
                                       
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    #需要更改
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 

            for key in list(state_dict.keys()):                
                if 'bert' in key:
                    new_key = key.replace('bert.','')
                    state_dict[new_key] = state_dict[key] 
                    del state_dict[key]
                
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch= config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']


    print("Start training")
    start_time = time.time()

    if not args.evaluate:
        if args.distributed:
            train_loader_hateful.sampler.set_epoch(max_epoch)
        train(model, train_loader_hateful, optimizer, tokenizer, warmup_steps, device, lr_scheduler, config, start_time, model_without_ddp, val_seen_loader_hateful, test_seen_loader_hateful, args)
        #train_stats = 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/workspace/project/ALBEF/configs/Hateful_mmd.yaml')
    parser.add_argument('--output_dir', default='/workspace/project/ALBEF/output/Hateful_cckt_all')  
    parser.add_argument('--checkpoint', default='/workspace/project/ALBEF/output/Hateful/ALBEF.pth')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
#   例如 python train.py --eval  那么你用了这个eval 那这个eval就是true
#   如果 python train.py   你没有用那个 eval 此时 eval 为false
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    #将hateful配置文件输出一份到output
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)







