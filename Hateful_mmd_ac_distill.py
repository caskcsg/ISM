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

from models.model_mmd_ac_distill import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from sklearn.metrics import roc_auc_score
import metrics
from itertools import cycle
from tqdm import tqdm

from module.dan import MultipleKernelMaximumMeanDiscrepancy
from module.kernels import GaussianKernel
import random


os.environ["CUDA_VISIBLE_DEVICES"] = "1"




mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)], 
        linear=False, quadratic_program=False
    )#在实际应用中，高斯核的σ \sigmaσ会取多个值，分别求核函数然后取和，作为最后的核函数


def cl_w(y_softmax, label, d):
    # y_softmax
    # tensor([[0.2109, 0.7891],
    #     [0.1750, 0.8250],
    #     [0.2597, 0.7403],
    #     [0.2826, 0.7174],
    #     [0.2379, 0.7621],
    #     [0.2669, 0.7331],
    #     [0.4190, 0.5810],
    #     [0.2277, 0.7723]], device='cuda:0', grad_fn=<SplitBackward>)
    the_index = torch.LongTensor(np.array(range(y_softmax.shape[0]))).cuda()  #[0, 1, 2, 3, 4, 5, 6, 7]
    #print(the_index.shape) #[8]   label: [0, 1, 0, 0, 0, 1, 0, 0]
    y_label = y_softmax[the_index, label] #tensor([0.2109, 0.8250, 0.2597, 0.2826, 0.2379, 0.7331, 0.4190, 0.2277],device='cuda:0', grad_fn=<PermuteBackward>)
    #print(y_label)
    
    weight_var = (y_label > d).float().detach()

    #print(weight_var)#[0, 1, 0, 0, 0, 1, 0, 0]
    source_weight = weight_var.data.clone()#[0, 1, 0, 0, 0, 1, 0, 0]
    
    # source_weight1,source_weight0=source_weight.chunk(2,0)
    # source_num_1 = float((torch.sum(source_weight1)))
    # source_num_0 = float((torch.sum(source_weight0)))
    
    return source_weight #, source_num_1,source_num_0


def train(model, data_loader_hateful, data_loader_twitter, optimizer, tokenizer, warmup_steps, device, scheduler, mkmmd_loss, config, start_time, model_without_ddp, val_seen_loader_hateful, val_unseen_loader_hateful, test_seen_loader_hateful, test_unseen_loader_hateful, args):
    
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”
    
    
    lr_scheduler = scheduler
    # train
    model.train()  

    mkmmd_loss.train()
    label_criterion = nn.CrossEntropyLoss().cuda()  #Default: ``'mean'`` 输出的总和将除以输出中的元素数
    weight_label_criterion = nn.CrossEntropyLoss(reduction='none').cuda()   #none'表示直接返回n份样本的loss(相当于reduce=False)
    len_data_loader_hateful = len(data_loader_hateful) - 1
    len_data_loader_twitter = len(data_loader_twitter) - 1

    best_seen = 0
    best_unseen = 0
    best_seen_epoch = 0
    best_unseen_epoch = 0
    
    best_seen_auc = 0
    best_unseen_auc = 0
    best_seen_auc_epoch = 0
    best_unseen_auc_epoch = 0

    best_seen_test_auc = 0
    best_unseen_test_auc = 0
    best_seen_test_auc_epoch = 0
    best_unseen_test_auc_epoch = 0  

    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    #header = 'Train Epoch: [{}]'.format(epoch)
    #print_freq = 50   
    step_size = 1000
    #warmup_iterations = warmup_steps*step_size  
 
    #for i,(images, text, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    #for i,(hateful_data, twitter_data) in enumerate(metric_logger.log_every(zip(cycle(data_loader_hateful), data_loader_twitter), print_freq, header, config)):
    #for i,(twitter_images, twitter_text, twitter_targets) in enumerate(metric_logger.log_every(data_loader_twitter, print_freq, header)):

    for i in range(0, config['schedular']['epochs']):

        if i % len_data_loader_twitter == 0:
            data_twitter = iter(data_loader_twitter)
        twitter_images, twitter_text, twitter_targets = next(data_twitter)

        
        if i % len_data_loader_hateful == 0:
            data_hateful = iter(data_loader_hateful)
        hateful_images, hateful_text, hateful_targets = next(data_hateful)

        images = torch.cat((hateful_images, twitter_images), 0)
        hateful_text.extend(twitter_text)
        targets = torch.cat((hateful_targets, twitter_targets), 0)


        images, targets = images.to(device,non_blocking=True), targets.to(device,non_blocking=True)

        text_inputs = tokenizer(hateful_text, padding='longest', return_tensors="pt").to(device) 

        if (i/1000)>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/(config['batch_size_train'] * 2))

        output, loss_distill = model(images, text_inputs, targets=targets, train=True, alpha=alpha)  
        #hateful_hidden_state,  twitter_hidden_state= output['last_hidden_state'].chunk(2, 0)
        hateful_image_embedding, twitter_image_embedding = output['image_embed'].chunk(2, 0)
        hateful_text_embedding, twitter_text_embedding = output['text_embed'].chunk(2, 0)


        #mmd_loss = mkmmd_loss(hateful_hidden_state, twitter_hidden_state)
        mmd_loss_image = mkmmd_loss(hateful_image_embedding, twitter_image_embedding)
        mmd_loss_text = mkmmd_loss(hateful_text_embedding, twitter_text_embedding)
        Ld = mmd_loss_image + mmd_loss_text


        hateful_head_hateful, twitter_head_hateful = output['head_hateful'].chunk(2, 0)
        hateful_head_hateful_softmax, twitter_head_hateful_softmax = output['head_hateful_softmax'].chunk(2, 0)
        ############改
        hateful_head_twitter, twitter_head_twitter = output['head_twitter'].chunk(2, 0)
        hateful_head_twitter_softmax, twitter_head_twitter_softmax = output['head_twitter_softmax'].chunk(2, 0)
        hateful_target, twitter_target = targets.chunk(2, 0)
        

        Ly_target = label_criterion(hateful_head_hateful, hateful_target)

        if i < 2500:
           Ly_source = label_criterion(twitter_head_twitter, twitter_target)
           #Ly = Ly_target + 0.1*Ly_source
           Ly = Ly_target + Ly_source
        else:
            # source_y_h 8个1、8个0label 和 torch.cat([label_var1,label_var0])计算权重
            # source_y_softmax_h 源样本过target分类器的softmax值
        #print(twitter_head_hateful_softmax.shape, twitter_target.shape) #[8, 2] [8]

        
            weight_source = cl_w(twitter_head_hateful_softmax, twitter_target, 0.8)
            
            loss_Ly_source = weight_label_criterion(twitter_head_twitter, twitter_target)
            Ly_source = (loss_Ly_source*weight_source).mean()
            # Ly = Ly_target + 0.8 * Ly_source
            Ly = Ly_target + Ly_source

        #Ly_source = label_criterion(twitter_head_twitter, twitter_target)
        #Ly = Ly_target + Ly_source
        
        loss = Ly  + 0.7 * Ld +  loss_distill
        #loss = Ly  +  Ld

        if i % 500 == 0:
            print('i={}  Ly_target={:.5f}   Ly_source={:.5f}   mmd_loss={:.5f}   sum_loss={:.5f}'.format(i, Ly_target, Ly_source, Ld, loss))
            
        optimizer.zero_grad()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if i%1000==0 and i%step_size==0 and i<=config['schedular']['epochs']: 
            scheduler.step(i//step_size)    

        
    # gather the stats from all processes
        if i % 500 == 0:
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger.global_avg())     

            train_stats = {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    

            val_seen_stats = evaluate(model, val_seen_loader_hateful, tokenizer, device, config)
            val_unseen_stats = evaluate(model, val_unseen_loader_hateful, tokenizer, device, config)
            #val_twitter_stats = evaluate(model, val_loader_twitter, tokenizer, device, config)

            test_seen_stats = evaluate(model, test_seen_loader_hateful, tokenizer, device, config)
            test_unseen_stats = evaluate(model, test_unseen_loader_hateful, tokenizer, device, config)
            #test_twitter_stats = evaluate(model, test_loader_twitter, tokenizer, device, config)

            if utils.is_main_process():  
                if args.evaluate:
                    log_stats = {**{f'val_seen_{k}': v for k, v in val_seen_stats.items()},
                                **{f'val_unseen_{k}': v for k, v in val_unseen_stats.items()},
                                **{f'test_seen_{k}': v for k, v in test_seen_stats.items()},
                                **{f'test_unseen_{k}': v for k, v in test_unseen_stats.items()},
                                'step': i,
                                }

                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")                
                else:    
                    log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'val_seen_{k}': v for k, v in val_seen_stats.items()},
                                **{f'val_unseen_{k}': v for k, v in val_unseen_stats.items()},
                                **{f'test_seen_{k}': v for k, v in test_seen_stats.items()},
                                **{f'test_unseen_{k}': v for k, v in test_unseen_stats.items()},
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

                    if float(val_unseen_stats['acc'])>best_unseen:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': i,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_unseen_best_acc.pth')) 
                        best_unseen = float(val_unseen_stats['acc'])
                        best_unseen_epoch=i

                    if float(val_seen_stats['auc_roc'])>best_seen_auc:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': i,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_seen_auc.pth')) 
                        best_seen_auc = float(val_seen_stats['auc_roc'])
                        best_seen_auc_epoch = i

                    if float(val_unseen_stats['auc_roc'])>best_unseen_auc:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': i,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_val_unseen_auc.pth')) 
                        best_unseen_auc = float(val_unseen_stats['auc_roc'])
                        best_unseen_auc_epoch = i

                    if float(test_seen_stats['auc_roc'])>best_seen_test_auc:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': i,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_test_seen_auc.pth')) 
                        best_seen_test_auc = float(test_seen_stats['auc_roc'])
                        best_seen_test_auc_epoch = i

                    if float(test_unseen_stats['auc_roc'])>best_unseen_test_auc:
                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'config': config,
                            'step': i,
                        }
                        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best_test_unseen_auc.pth')) 
                        best_unseen_test_auc = float(test_unseen_stats['auc_roc'])
                        best_unseen_test_auc_epoch = i

            if args.evaluate:
                break
            epoch=i/500
            lr_scheduler.step(epoch+warmup_steps+1)  
            dist.barrier()   
                    
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str)) 
                
            if utils.is_main_process():   
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write("best val seen acc step: %d"%best_seen_epoch)         
                    f.write("best val unseen acc step: %d"%best_unseen_epoch)         
                    f.write("best val seen auc step: %d"%best_seen_auc_epoch)
                    f.write("best val unseen auc step: %d"%best_unseen_auc_epoch)
                    f.write("best test seen auc step: %d"%best_seen_test_auc_epoch)
                    f.write("best test unseen auc step: %d"%best_unseen_test_auc_epoch)

            model.train()


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
    
    # sklearn中的auc_roc计算
    # pre_list = F.softmax(pre_list, dim=-1)[:, 1]#[size, 1]
    # label_list = label_list.unsqueeze(1) 
    # auc_roc = roc_auc_score(label_list.cpu().numpy(), pre_list.cpu().numpy())

    # metrics中auc_roc计算
    auc_roc = metrics.ROC_AUC()(pre_list, label_list)
    auc_roc = auc_roc.item()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    print("Averaged stats: Auc_Roc: ", auc_roc)  
    score_dic = {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    score_dic['auc_roc'] = auc_roc
    return score_dic
    
    
def main(args, config):
    utils.init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    #需要更改
    datasets = create_dataset('cckt', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, True, False, False, False, False, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None, None, None, None, None, None]

    train_loader_hateful, train_loader_twitter, val_seen_loader_hateful, val_unseen_loader_hateful, val_loader_twitter, test_seen_loader_hateful, test_unseen_loader_hateful, test_loader_twitter = create_loader(
                                                          datasets,samplers,
                                                          batch_size=[config['batch_size_train']]*2+[config['batch_size_test']]*6,
                                                          num_workers=[4,4,4,4,4,4,4,4],is_trains=[True,True,False,False,False,False,False,False], 
                                                          collate_fns=[None,None,None,None,None,None,None,None]
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
            train_loader_twitter.sampler.set_epoch(max_epoch)
        train(model, train_loader_hateful, train_loader_twitter, optimizer, tokenizer,  warmup_steps, device, lr_scheduler, mkmmd_loss, config, start_time, model_without_ddp, val_seen_loader_hateful, val_unseen_loader_hateful, test_seen_loader_hateful, test_unseen_loader_hateful, args)
        #train_stats = 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/workspace/project/ALBEF/configs/Hateful_mmd_ac_distill.yaml')
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




