from functools import partial
from models.vit_6 import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F



class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']
        self.config = config

        ###
        embed_dim = config['embed_dim']  #256
        vision_width = config['vision_width']#768


        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)          


        ###
        self.temp = nn.Parameter(torch.ones([]) * config['temp']) #0.07
        text_width = self.text_encoder.config.hidden_size #768
        #hateful
        self.hateful_vision_proj = nn.Linear(vision_width, embed_dim) #[768, 256]
        self.hateful_text_proj = nn.Linear(text_width, embed_dim)  #[768, 256]
        #twitter
        self.twitter_vision_proj = nn.Linear(vision_width, embed_dim) #[768, 256]
        self.twitter_text_proj = nn.Linear(text_width, embed_dim)  #[768, 256]
        #fusion
        ####
        # self.vision_proj = nn.Linear(vision_width, embed_dim) #[768, 256]
        # self.text_proj = nn.Linear(text_width, embed_dim)  #[768, 256]
        ####
        self.queue_size = config['queue_size'] #65536


        self.cls_head_hateful = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)#ve是三分类需要改成二分类
                )
        ##################改
        self.cls_head_twitter = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)#ve是三分类需要改成二分类
                )
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.cls_head_hateful[0].weight.data)
        nn.init.kaiming_normal_(self.cls_head_hateful[2].weight.data)
        nn.init.kaiming_normal_(self.cls_head_twitter[0].weight.data)
        nn.init.kaiming_normal_(self.cls_head_twitter[2].weight.data)


        ### create the queue
        #Hateful
        self.register_buffer("hateful_image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("hateful_text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("hateful_queue_ptr", torch.zeros(1, dtype=torch.long))  
        #Twitter
        self.register_buffer("twitter_image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("twitter_text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("twitter_queue_ptr", torch.zeros(1, dtype=torch.long)) 
        #fusion
        ####
        # self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
        ####
        #hateful
        self.hateful_image_queue = nn.functional.normalize(self.hateful_image_queue, dim=0)
        self.hateful_text_queue = nn.functional.normalize(self.hateful_text_queue, dim=0)
        #twitter
        self.twitter_image_queue = nn.functional.normalize(self.twitter_image_queue, dim=0)
        self.twitter_text_queue = nn.functional.normalize(self.twitter_text_queue, dim=0)
        #fusion
        ####
        # self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        ###

        #不用distill
        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))               
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
            self.cls_head_hateful_m = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)#ve是三分类需要改成二分类
                )

            self.cls_head_twitter_m = nn.Sequential(
                    nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.text_encoder.config.hidden_size, 2)#ve是三分类需要改成二分类
                    )

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.cls_head_hateful,self.cls_head_hateful_m],
                                [self.cls_head_twitter,self.cls_head_twitter_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995

        #hateful
        self.hateful_vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.hateful_text_proj_m = nn.Linear(text_width, embed_dim)
        #twitter
        self.twitter_vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.twitter_text_proj_m = nn.Linear(text_width, embed_dim)
        #fusion
        ####
        # self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        # self.text_proj_m = nn.Linear(text_width, embed_dim)
        ####
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds, image_hidden_state = self.visual_encoder(image) 

        #print('image_embeds:', image_embeds.size()) #[20, 577, 768]
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device) 
        #print('image_atts:',  image_atts.size())      #[20, 577] 

        
        if train:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      ) 

            image_embed = image_embeds[:,0,:]
            hateful_image_embed, twitter_image_embed = image_embed.chunk(2, 0)
            cdan_image_embed = image_hidden_state[5][:,0,:]
            text_embed = output.hidden_states[6][:,0,:]
            last_hidden_state = output.last_hidden_state[:,0,:]
            #print("last_hidden_state:", last_hidden_state.size()) #[20, 768]
            # output.hidden_states type:tuple

            # hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，len() = 13
            # 它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
            #print(output.hidden_states[7][:,0,:].size()) #output.hidden_states[1].size() torch.Size([20, 32, 768])
            
            head_hateful = self.cls_head_hateful(last_hidden_state)
            head_hateful_softmax = self.softmax(head_hateful)
            head_twitter = self.cls_head_twitter(last_hidden_state) 
            head_twitter_softmax = self.softmax(head_twitter)
            hateful_head_hateful, twitter_head_hateful = head_hateful.chunk(2, 0)
            hateful_head_twitter, twitter_head_twitter = head_twitter.chunk(2, 0)

            prediction = torch.cat((hateful_head_hateful, twitter_head_twitter), dim=0)
            ###
            #hateful_embed, twitter_embed = output.hidden_states[7][:,0,:].chunk(2, 0)
            ###

            hateful_image_feat = F.normalize(self.hateful_vision_proj(hateful_image_embed),dim=-1) #256
            twitter_image_feat = F.normalize(self.twitter_vision_proj(twitter_image_embed),dim=-1) #256
            #fusion
            ###
            # image_feat = F.normalize(self.vision_proj(hateful_embed),dim=-1)
            # text_feat = F.normalize(self.text_proj(twitter_embed),dim=-1)
            ###

            ###单一
            # hateful_image_feat_hateful, twitter_image_feat_hateful = hateful_image_feat.chunk(2, 0)
            # hateful_image_feat_twitter, twitter_image_feat_twitter = twitter_image_feat.chunk(2, 0)
            hateful_text_embed, twitter_text_embed = output.hidden_states[6][:,0,:].chunk(2, 0)
            hateful_text_feat = F.normalize(self.hateful_text_proj(hateful_text_embed),dim=-1) #256
            twitter_text_feat = F.normalize(self.twitter_text_proj(twitter_text_embed),dim=-1) #256

            ###单一
            # hateful_text_feat_hateful, twitter_text_feat_hateful = hateful_text_feat.chunk(2, 0)
            # hateful_text_feat_twitter, twitter_text_feat_twitter = twitter_text_feat.chunk(2, 0)

            if self.distill:                
                with torch.no_grad():
                    self.temp.clamp_(0.001,0.5)
                    self._momentum_update()
                    image_embeds_m, image_hidden_state_m = self.visual_encoder_m(image) 
                    output_m = self.text_encoder_m(text.input_ids, 
                                                attention_mask = text.attention_mask, 
                                                encoder_hidden_states = image_embeds_m,
                                                encoder_attention_mask = image_atts,        
                                                return_dict = True
                                                )      

                    head_hateful_m = self.cls_head_hateful_m(output_m.last_hidden_state[:,0,:]) 
                    hateful_head_hateful_m, twitter_head_hateful_m = head_hateful_m.chunk(2, 0)
                    head_twitter_m = self.cls_head_twitter_m(output_m.last_hidden_state[:,0,:]) 
                    hateful_head_twitter_m, twitter_head_twitter_m = head_twitter_m.chunk(2, 0)
                    prediction_m = torch.cat((hateful_head_hateful_m, twitter_head_twitter_m), dim=0)

                    #fusion
                    #####
                    # hateful_embed_m, twitter_embed_m = output_m.hidden_states[7][:,0,:].chunk(2, 0)
                    # image_feat_m = F.normalize(self.vision_proj_m(hateful_embed_m),dim=-1) 
                    # image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)  
                    # text_feat_m = F.normalize(self.text_proj_m(twitter_embed_m),dim=-1) 
                    # text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
                    # sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
                    # sim_t2i_m = text_feat_m @ image_feat_all / self.temp 
                    # sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                    # sim_targets.fill_diagonal_(1)  
                    # sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                    # sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets 
                    #####
                    hateful_image_embed_m, twitter_image_embed_m = image_embeds_m[:,0,:].chunk(2, 0)
                    hateful_image_feat_m = F.normalize(self.hateful_vision_proj_m(hateful_image_embed_m),dim=-1)  
                    twitter_image_feat_m = F.normalize(self.twitter_vision_proj_m(twitter_image_embed_m),dim=-1)  

                    ###单一
                    # hateful_image_feat_m_hateful, twitter_image_feat_m_hateful = hateful_image_feat_m.chunk(2, 0)
                    # hateful_image_feat_m_twitter, twitter_image_feat_m_twitter = twitter_image_feat_m.chunk(2, 0)

                    hateful_image_feat_all = torch.cat([hateful_image_feat_m.t(),self.hateful_image_queue.clone().detach()],dim=1)
                    twitter_image_feat_all = torch.cat([twitter_image_feat_m.t(),self.twitter_image_queue.clone().detach()],dim=1)                                         
                                         
                    text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                        return_dict = True, mode = 'text')  
                    hateful_text_embed_m, twitter_text_embed_m = text_output_m.last_hidden_state[:,0,:].chunk(2, 0)
                    hateful_text_feat_m = F.normalize(self.hateful_text_proj_m(hateful_text_embed_m),dim=-1) 
                    twitter_text_feat_m = F.normalize(self.twitter_text_proj_m(twitter_text_embed_m),dim=-1) 

                    ###单一
                    # hateful_text_feat_m_hateful, twitter_text_feat_m_hateful = hateful_text_feat_m.chunk(2, 0)
                    # hateful_text_feat_m_twitter, twitter_text_feat_m_twitter = twitter_text_feat_m.chunk(2, 0)

                    hateful_text_feat_all = torch.cat([hateful_text_feat_m.t(),self.hateful_text_queue.clone().detach()],dim=1)
                    twitter_text_feat_all = torch.cat([twitter_text_feat_m.t(),self.twitter_text_queue.clone().detach()],dim=1)

                    hateful_sim_i2t_m = hateful_image_feat_m @ hateful_text_feat_all / self.temp 
                    hateful_sim_t2i_m = hateful_text_feat_m @ hateful_image_feat_all / self.temp 

                    twitter_sim_i2t_m = twitter_image_feat_m @ twitter_text_feat_all / self.temp 
                    twitter_sim_t2i_m = twitter_text_feat_m @ twitter_image_feat_all / self.temp

                    hateful_sim_targets = torch.zeros(hateful_sim_i2t_m.size()).to(image.device)
                    twitter_sim_targets = torch.zeros(twitter_sim_i2t_m.size()).to(image.device)

                    hateful_sim_targets.fill_diagonal_(1)  
                    twitter_sim_targets.fill_diagonal_(1)  


                    hateful_sim_i2t_targets = alpha * F.softmax(hateful_sim_i2t_m, dim=1) + (1 - alpha) * hateful_sim_targets
                    hateful_sim_t2i_targets = alpha * F.softmax(hateful_sim_t2i_m, dim=1) + (1 - alpha) * hateful_sim_targets 

                    twitter_sim_i2t_targets = alpha * F.softmax(twitter_sim_i2t_m, dim=1) + (1 - alpha) * twitter_sim_targets
                    twitter_sim_t2i_targets = alpha * F.softmax(twitter_sim_t2i_m, dim=1) + (1 - alpha) * twitter_sim_targets 

                hateful_sim_i2t = hateful_image_feat @ hateful_text_feat_all / self.temp 
                hateful_sim_t2i = hateful_text_feat @ hateful_image_feat_all / self.temp 

                twitter_sim_i2t = twitter_image_feat @ twitter_text_feat_all / self.temp 
                twitter_sim_t2i = twitter_text_feat @ twitter_image_feat_all / self.temp 

                hateful_loss_i2t = -torch.sum(F.log_softmax(hateful_sim_i2t, dim=1)*hateful_sim_i2t_targets,dim=1).mean()
                hateful_loss_t2i = -torch.sum(F.log_softmax(hateful_sim_t2i, dim=1)*hateful_sim_t2i_targets,dim=1).mean()

                twitter_loss_i2t = -torch.sum(F.log_softmax(twitter_sim_i2t, dim=1)*twitter_sim_i2t_targets,dim=1).mean()
                twitter_loss_t2i = -torch.sum(F.log_softmax(twitter_sim_t2i, dim=1)*twitter_sim_t2i_targets,dim=1).mean() 

                hateful_loss_ita = (hateful_loss_i2t+hateful_loss_t2i)/2  * 0.1
                twitter_loss_ita = (twitter_loss_i2t+twitter_loss_t2i)/2  * 0.1
                ##fusion
                ####
                # sim_i2t = image_feat @ text_feat_all / self.temp 
                # sim_t2i = text_feat @ image_feat_all / self.temp 
                # loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
                # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
                # loss_ita = (loss_i2t+loss_t2i)/2  
                #self._dequeue_and_enqueue(image_feat_m, text_feat_m)
                ####
                self._dequeue_and_enqueue_hateful(hateful_image_feat_m, hateful_text_feat_m)
                self._dequeue_and_enqueue_twitter(twitter_image_feat_m, twitter_text_feat_m)

                loss_fc = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                    F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()

                output = {
                            'last_hidden_state': last_hidden_state,
                            'head_hateful': head_hateful,
                            'head_hateful_softmax': head_hateful_softmax,
                            'head_twitter': head_twitter,
                            'head_twitter_softmax': head_twitter_softmax,
                            'image_embed': cdan_image_embed,
                            'text_embed': text_embed,
                        }

            else:
                output = None
                hateful_loss_ita = None
                twitter_loss_ita = None
                loss_fc = F.cross_entropy(prediction, targets)                
            return output, loss_fc, hateful_loss_ita, twitter_loss_ita
            
        else:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )         
            
            prediction_hateful = self.cls_head_hateful(output.last_hidden_state[:,0,:])          
              
            return prediction_hateful
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()
    def _dequeue_and_enqueue_hateful(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.hateful_queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.hateful_image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.hateful_text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.hateful_queue_ptr[0] = ptr 

    @torch.no_grad()
    def _dequeue_and_enqueue_twitter(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.twitter_queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.twitter_image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.twitter_text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.twitter_queue_ptr[0] = ptr 

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


