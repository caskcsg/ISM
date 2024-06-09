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
        self.vision_proj = nn.Linear(vision_width, embed_dim) #[768, 256]
        self.text_proj = nn.Linear(text_width, embed_dim)  #[768, 256]
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

        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
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
            cdan_image_embed = image_hidden_state[5][:,0,:]
            text_embed = output.hidden_states[6][:,0,:]
            last_hidden_state = output.last_hidden_state[:,0,:]
            
            head_hateful = self.cls_head_hateful(last_hidden_state)
            head_hateful_softmax = self.softmax(head_hateful)
            head_twitter = self.cls_head_twitter(last_hidden_state) 
            head_twitter_softmax = self.softmax(head_twitter)
            hateful_head_hateful, twitter_head_hateful = head_hateful.chunk(2, 0)
            hateful_head_twitter, twitter_head_twitter = head_twitter.chunk(2, 0)

            prediction = torch.cat((hateful_head_hateful, twitter_head_twitter), dim=0)

            hateful_embed_image, twitter_embed_image = cdan_image_embed.chunk(2, 0)
            hateful_embed_text, twitter_embed_text = text_embed.chunk(2, 0)

            image_feat = F.normalize(self.vision_proj(hateful_embed_image),dim=-1)
            text_feat = F.normalize(self.text_proj(hateful_embed_text),dim=-1)
              
            with torch.no_grad():
                self.temp.clamp_(0.001,0.5)

                image_feat_m = F.normalize(self.vision_proj_m(twitter_embed_image),dim=-1) 
                image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)  
                text_feat_m = F.normalize(self.text_proj_m(twitter_embed_text),dim=-1) 
                text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
                sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp 
                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets.fill_diagonal_(1)  
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets 
 
            sim_i2t = image_feat @ text_feat_all / self.temp 
            sim_t2i = text_feat @ image_feat_all / self.temp 
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 
            loss_ita = (loss_i2t+loss_t2i)/2 * 0.1
            self._dequeue_and_enqueue(image_feat_m, text_feat_m)

            loss_fc = F.cross_entropy(prediction, targets)

            output = {
                        'last_hidden_state': last_hidden_state,
                        'head_hateful': head_hateful,
                        'head_hateful_softmax': head_hateful_softmax,
                        'head_twitter': head_twitter,
                        'head_twitter_softmax': head_twitter_softmax,
                        'image_embed': image_embed,
                        'text_embed': text_embed,
                    }              
            return output, loss_fc, loss_ita
            
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

