from functools import partial
from models.vit_6 import VisionTransformer
from models.xbert import BertConfig, BertModel
from models.MultiHeadAttention import MultiHeadAttention
from models.MultiHeadAttention_hateful import MultiHeadAttention_hateful
from module.dan import MultipleKernelMaximumMeanDiscrepancy
from module.kernels import GaussianKernel


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 


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

        

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))  

        scale = 768 ** -0.5  
        self.proj_h = nn.Parameter(scale * torch.randn(768, 100))
        self.proj_t = nn.Parameter(scale * torch.randn(768, 100))
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
                kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)], 
                linear=False, quadratic_program=False
        )#在实际应用中，高斯核的σ \sigmaσ会取多个值，分别求核函数然后取和，作为最后的核函数



        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)       
        #self.selfAttention = BertSelfAttention(bert_config, True)
        self.attention = MultiHeadAttention(1, 768, 768, 768)
        self.attention_hateful = MultiHeadAttention_hateful(1, 768, 768, 768)
        self.text_projection_h = nn.Parameter(torch.empty(768, 100))   
        nn.init.normal_(self.text_projection_h, std=768 ** -0.5)
        self.text_projection_t = nn.Parameter(torch.empty(768, 100))   
        nn.init.normal_(self.text_projection_t, std=768 ** -0.5)


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

        # self.logit_scale_h = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        #不用distill
        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))               
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
            self.cls_head_m = nn.Sequential(
                      nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                      nn.ReLU(),
                      nn.Linear(self.text_encoder.config.hidden_size, 2)#同上
                    )

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.cls_head,self.cls_head_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995
            
            
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

            # hateful_image_embedding, twitter_image_embedding = image_embeds[:,0,:].chunk(2, 0)
            # hateful_text_embedding, twitter_text_embedding = output.hidden_states[7][:,0,:].chunk(2, 0)
            image_embed = image_embeds[:,0,:]
            text_embed = output.hidden_states[6][:,0,:]
            text_embed_t = output.hidden_states[6][:,0,:]
            #print(image_embed.size(), text_embed.size())
            encoder_embedding_hateful,  encoder_embedding_twitter= image_embeds.chunk(2, 0)
            hidden_states_hateful, hidden_states_twitter = output.last_hidden_state.chunk(2, 0)
            self_outputs_t, attn_t = self.attention(hidden_states_twitter, encoder_embedding_twitter, encoder_embedding_twitter)
            self_outputs_h, attn_h = self.attention_hateful(hidden_states_hateful, encoder_embedding_hateful, encoder_embedding_hateful)

            # self_outputs = self.selfAttention(
            #                     hidden_states_twitter,
            #                     None,
            #                     None,
            #                     encoder_embedding_twitter,
            #                     None,
            #                     None,
            #                     False,
            #                 )
            #print("last_hidden_state:", output.last_hidden_state.size())#[20, 32, 768]
            #print("last_hidden_state:", last_hidden_state.size()) #[20, 768]
            # output.hidden_states type:tuple
            
            hateful_i, twitter_i = image_embed.chunk(2, 0)
            h_i = hateful_i @ self.proj_h
            t_i = twitter_i @ self.proj_t
            hateful_t, twitter_t = text_embed_t.chunk(2, 0)
            h_t = hateful_t @ self.text_projection_h
            t_t = twitter_t @ self.text_projection_t

            hi = h_i / h_i.norm(dim=-1, keepdim=True)
            ht = h_t / h_t.norm(dim=-1, keepdim=True)
            ti = t_i / t_i.norm(dim=-1, keepdim=True)
            tt = t_t / t_t.norm(dim=-1, keepdim=True)

            kl_loss_h = self.mkmmd_loss(hi, ht)
            kl_loss_t = self.mkmmd_loss(ti, tt)
            print(kl_loss_h, kl_loss_t)



            # p_loss_h = F.kl_div(F.log_softmax(hi, dim=-1), F.softmax(ht, dim=-1), reduction='none')
            # q_loss_h = F.kl_div(F.log_softmax(ht, dim=-1), F.softmax(hi, dim=-1), reduction='none')
            # p_loss_t = F.kl_div(F.log_softmax(ti, dim=-1), F.softmax(tt, dim=-1), reduction='none')
            # q_loss_t = F.kl_div(F.log_softmax(tt, dim=-1), F.softmax(ti, dim=-1), reduction='none')
            # You can choose whether to use function "sum" and "mean" depending on your task
            # p_loss_h = p_loss_h.sum()
            # q_loss_h = q_loss_h.sum()
            # kl_loss_h = (p_loss_h + q_loss_h) / 2
            # p_loss_t = p_loss_t.sum()
            # q_loss_t = q_loss_t.sum()
            # kl_loss_t = (p_loss_t + q_loss_t) / 2
            if abs(kl_loss_h - kl_loss_t) >= 0.005:
                res = torch.cat((self_outputs_h, self_outputs_t), 0)
                output.last_hidden_state = res + output.last_hidden_state
            # hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，len() = 13
            # 它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
            #print(output.hidden_states[7][:,0,:].size()) #output.hidden_states[1].size() torch.Size([20, 32, 768])
            last_hidden_state = output.last_hidden_state[:,0,:]
            head_hateful = self.cls_head_hateful(last_hidden_state)
            head_hateful_softmax = self.softmax(head_hateful)
            head_twitter = self.cls_head_twitter(last_hidden_state) 
            head_twitter_softmax = self.softmax(head_twitter)
            output = {
                        'last_hidden_state': last_hidden_state,
                        'head_hateful': head_hateful,
                        'head_hateful_softmax': head_hateful_softmax,
                        'head_twitter': head_twitter,
                        'head_twitter_softmax': head_twitter_softmax,
                        'image_embed': image_embed,
                        'text_embed': text_embed,
                    }
            return output

            # if self.distill:                
            #     with torch.no_grad():
            #         self._momentum_update()
            #         image_embeds_m = self.visual_encoder_m(image) 
            #         output_m = self.text_encoder_m(text.input_ids, 
            #                                    attention_mask = text.attention_mask, 
            #                                    encoder_hidden_states = image_embeds_m,
            #                                    encoder_attention_mask = image_atts,        
            #                                    return_dict = True
            #                                   )           
            #         prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   
            #     loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
            #         F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            # else:
            #     loss_hateful = F.cross_entropy(prediction_hateful, target_hateful)   
            #     loss_twitter = F.cross_entropy(prediction_twitter, target_twitter)                

            # return loss_hateful,  loss_twitter
            
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
    def cos_sim(self, h_i, h_t, t_i, t_t):
        """
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a 
        :param vector_b: 向量 b
        :return: sim
        """
        hi = np.mat(h_i.cpu().numpy())
        ht = np.mat(h_t.cpu().numpy())
        ti = np.mat(t_i.cpu().numpy())
        tt = np.mat(t_t.cpu().numpy())
        num_h = hi * ht.T
        num_t = ti * tt.T
        denom_h = np.linalg.norm(hi) * np.linalg.norm(ht)
        denom_t = np.linalg.norm(ti) * np.linalg.norm(tt)
        cos_h = num_h / denom_h
        cos_t = num_t / denom_t
        sim_h = 0.5 + 0.5 * cos_h
        sim_t = 0.5 + 0.5 * cos_t
        print(sim_h)
        print(sim_t)
        # if abs(sim_h - sim_t) >= 0.2:
        #     if sim_h > sim_t:
        #         return np.repeat(sim_t.cpu().numpy(), 8)
        #     else :
        #         return np.repeat(sim_h.cpu().numpy(), 8)
        # else :
        #     a = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        #     return a
        

    @torch.no_grad()    
    def js(self, h_i, h_t, t_i, t_t):
        M_h = (h_i + h_t) / 2
        M_t = (t_i + t_t) / 2
        p_loss_h = F.kl_div(F.log_softmax(h_i, dim=-1), F.softmax(M_h, dim=-1), reduction='none')
        q_loss_h = F.kl_div(F.log_softmax(h_t, dim=-1), F.softmax(M_h, dim=-1), reduction='none')
        p_loss_t = F.kl_div(F.log_softmax(t_i, dim=-1), F.softmax(M_t, dim=-1), reduction='none')
        q_loss_t = F.kl_div(F.log_softmax(t_t, dim=-1), F.softmax(M_t, dim=-1), reduction='none')
        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss_h = p_loss_h.mean()
        q_loss_h = q_loss_h.mean()
        js_loss_h = (p_loss_h + q_loss_h) / 2
        p_loss_t = p_loss_t.mean()
        q_loss_t = q_loss_t.mean()
        js_loss_t = (p_loss_t + q_loss_t) / 2
        print(js_loss_h)
        print(js_loss_t)
        if abs(js_loss_h - js_loss_t) >= 0.1:
            if js_loss_h > js_loss_t:
                return np.repeat(js_loss_t.cpu().numpy(), 8)
            else :
                return np.repeat(js_loss_h.cpu().numpy(), 8)
        else :
            a = np.array([1, 1, 1, 1, 1, 1, 1, 1])
            return a


    @torch.no_grad()    
    def w(self, h_i, h_t, t_i, t_t):
        p_loss_h = F.kl_div(F.log_softmax(h_i, dim=-1), F.softmax(h_t, dim=-1), reduction='none')
        q_loss_h = F.kl_div(F.log_softmax(h_t, dim=-1), F.softmax(h_i, dim=-1), reduction='none')
        p_loss_t = F.kl_div(F.log_softmax(t_i, dim=-1), F.softmax(t_t, dim=-1), reduction='none')
        q_loss_t = F.kl_div(F.log_softmax(t_t, dim=-1), F.softmax(t_i, dim=-1), reduction='none')
        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss_h = p_loss_h.sum()
        q_loss_h = q_loss_h.sum()
        kl_loss_h = (p_loss_h + q_loss_h) / 2
        p_loss_t = p_loss_t.sum()
        q_loss_t = q_loss_t.sum()
        kl_loss_t = (p_loss_t + q_loss_t) / 2
        print(kl_loss_h, kl_loss_t)
        if abs(kl_loss_h - kl_loss_t) >= 0.002:
            if kl_loss_h > kl_loss_t:
                return np.repeat(kl_loss_t.cpu().numpy(), 8)
            else :
                return np.repeat(kl_loss_h.cpu().numpy(), 8)
        else :
            a = np.array([1, 1, 1, 1, 1, 1, 1, 1])
            return a

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
                


