from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from Hateful_graph_final.graph_proj import Classifier_proj
from Hateful_graph_final.graph_classifier  import Classifier
from Hateful_graph_final.DeepCCA import cca_loss
import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np

# from module.latent import Latent_Bert, Latent_Bert_H, Latent_Bert_T


def bce_for_loss(logits,labels):
    loss=nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss*=labels.size(1)
    return loss

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        self.tokenizer = tokenizer 
        self.distill = config['distill']
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)          
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)#ve是三分类需要改成二分类
                )
        ##是否去掉投影层
        # self.project_t = nn.Sequential()
        # self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=768))
        # self.project_t.add_module('project_t_activation', nn.ReLU())
        # self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(768))

        # self.project_v = nn.Sequential()
        # self.project_v.add_module('project_v', nn.Linear(in_features=768, out_features=768))
        # self.project_v.add_module('project_v_activation', nn.ReLU())
        # self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(768))


        # self.weights = torch.nn.Parameter(torch.ones(2).float())


        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=self.text_encoder.config.hidden_size, out_features=self.text_encoder.config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=self.text_encoder.config.hidden_size, out_features=self.text_encoder.config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=self.text_encoder.config.hidden_size, out_features=self.text_encoder.config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())
        # self.proj_cls = nn.Linear(256, 768)
        self.classifier_proj = Classifier_proj(768, 384)  #512-> 384
        self.gat = dglnn.GATConv(384, 256, num_heads=1, feat_drop=0.4, activation=F.elu)
        self.classifier = Classifier(256, 128)
        self.proj = nn.Linear(1024, 768)  ###1792

        self.cca = cca_loss(128, True)
        # self.proj_a = nn.Linear(768, 128)
        # self.dis_H = Latent_Bert_H()   
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # params = torch.ones(2, requires_grad=True)
        # self.params = torch.nn.Parameter(params)

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
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


    def compute_kl_loss(self, p, q):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        kl_loss = (p_loss + q_loss) / 2
        return kl_loss

    def shared_private(self, utterance_t, utterance_v):
        
        # Projecting to same sized space
        # utterance_t = self.project_t(utterance_t)
        # utterance_v = self.project_v(utterance_v)

        # Private-shared components
        utt_private_t = self.private_t(utterance_t)
        utt_private_v = self.private_v(utterance_v)

        utt_shared_t = self.shared(utterance_t)
        utt_shared_v = self.shared(utterance_v)

        return utt_private_t, utt_private_v, utt_shared_t, utt_shared_v

    def private_graph(self, private_image_embeds, private_output):
        batch = private_image_embeds.size()[0]    
        # image_node = image_embeds.size()[1] - 1
        image_node = private_image_embeds.size()[1]

        text_node = private_output.size()[1]
        g_i = [] 
        for i in range(batch):
            u_i= []
            v_i = []
            for j in range(image_node):
                u_i.append([j]*(image_node-1))
                for k in range(image_node):
                    if k != j:
                        v_i.append(k)
            #print(v_i, u_i)
            u_i = [n for a in u_i for n in a]
            g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(private_image_embeds.device)
            # g1.ndata['x'] = image_embeds[i,1:,:]
            g1.ndata['x'] = private_image_embeds[i,:,:]

            g_i.append(g1)
        bg_i = dgl.batch(g_i)  ####批量图像图

        g_t = []
        for i in range(batch):
            u_i= []
            v_i = []
            for j in range(text_node):
                u_i.append([j]*(text_node-1))
                for k in range(text_node):
                    if k != j:
                        v_i.append(k)
            #print(v_i, u_i)
            u_i = [n for a in u_i for n in a]
            g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(private_image_embeds.device)
            g1.ndata['x'] = private_output[i,:,:]
            # g1.ndata['x'] = output.hidden_states[6][i,1:,:]

            g_t.append(g1)
        bg_t = dgl.batch(g_t)  ####批量文本图

        feat_i = bg_i.ndata['x']
        feat_t = bg_t.ndata['x']
        proj_i = self.classifier_proj(bg_i, feat_i)
        proj_t = self.classifier_proj(bg_t, feat_t)

        # with bg_i.local_scope():
        #     bg_i.ndata['x'] = proj_i
        #     hg_i = dgl.mean_nodes(bg_i, 'x')
        #     bg_t.ndata['x'] = proj_t
        #     hg_t = dgl.mean_nodes(bg_t, 'x')
        # # print(hg_i.shape)
        # private_loss = self.compute_kl_loss(hg_i, hg_t)


        #####利用投影后的节点值构建多模态图
        g_f = []
        for i in range(batch):
            u_i= []
            v_i = []
            for j in range(image_node+text_node):
                u_i.append([j]*(image_node+text_node-1))
                for k in range(image_node+text_node):
                    if k != j:
                        v_i.append(k)
            u_i = [n for a in u_i for n in a]
            g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(private_image_embeds.device)
            g_f.append(g1)
        bg_f = dgl.batch(g_f)  ####批量融合图
        bg_f.ndata['x'] = torch.cat((proj_i, proj_t), dim=0)

        feat_f = bg_f.ndata['x']
        gat_f = self.gat(bg_f, feat_f)
        private_graph_embedding = self.classifier(bg_f, gat_f)  ###[B,128]
        

        return private_graph_embedding#, private_loss

    def share_graph(self, share_image_embeds, share_output):

        batch = share_image_embeds.size()[0] 
        image_node = share_image_embeds.size()[1]

        text_node = share_output.size()[1]
        g_i = [] 
        for i in range(batch):
            u_i= []
            v_i = []
            for j in range(image_node):
                u_i.append([j]*(image_node-1))
                for k in range(image_node):
                    if k != j:
                        v_i.append(k)
            #print(v_i, u_i)
            u_i = [n for a in u_i for n in a]
            g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(share_image_embeds.device)
            # g1.ndata['x'] = image_embeds[i,1:,:]
            g1.ndata['x'] = share_image_embeds[i,:,:]

            g_i.append(g1)
        bg_i = dgl.batch(g_i)  ####批量图像图

        g_t = []
        for i in range(batch):
            u_i= []
            v_i = []
            for j in range(text_node):
                u_i.append([j]*(text_node-1))
                for k in range(text_node):
                    if k != j:
                        v_i.append(k)
            #print(v_i, u_i)
            u_i = [n for a in u_i for n in a]
            g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(share_image_embeds.device)
            g1.ndata['x'] = share_output[i,:,:]
            # g1.ndata['x'] = output.hidden_states[6][i,1:,:]

            g_t.append(g1)
        bg_t = dgl.batch(g_t)  ####批量文本图

        feat_i = bg_i.ndata['x']
        feat_t = bg_t.ndata['x']
        proj_i = self.classifier_proj(bg_i, feat_i)
        proj_t = self.classifier_proj(bg_t, feat_t)

        # ###res
        with bg_i.local_scope():
            bg_i.ndata['x'] = proj_i
            hg_i = dgl.mean_nodes(bg_i, 'x')
            bg_t.ndata['x'] = proj_t
            hg_t = dgl.mean_nodes(bg_t, 'x')
        
        share_loss = self.compute_kl_loss(hg_i, hg_t)

        #####利用投影后的节点值构建多模态图
        g_f = []
        for i in range(batch):
            u_i= []
            v_i = []
            for j in range(image_node+text_node):
                u_i.append([j]*(image_node+text_node-1))
                for k in range(image_node+text_node):
                    if k != j:
                        v_i.append(k)
            u_i = [n for a in u_i for n in a]
            g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(share_image_embeds.device)
            g_f.append(g1)
        bg_f = dgl.batch(g_f)  ####批量融合图
        bg_f.ndata['x'] = torch.cat((proj_i, proj_t), dim=0)

        feat_f = bg_f.ndata['x']
        gat_f = self.gat(bg_f, feat_f)
        share_graph_embedding = self.classifier(bg_f, gat_f)  ###[B,128]
        return share_graph_embedding, share_loss
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        bt = image_embeds.size()[0]

        if train:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )  

            utt_private_t, utt_private_v, utt_shared_t, utt_shared_v = self.shared_private(output.hidden_states[7], image_embeds) 
            private_graph_embedding = self.private_graph(utt_private_v, utt_private_t)
            share_graph_embedding, share_loss = self.share_graph(utt_shared_v, utt_shared_t)
            corr = self.cca.loss(private_graph_embedding[:,0,:], share_graph_embedding[:,0,:], image.device)
            # print(corr)

            # weight = self.cos(share_graph_embedding[:,0,:], private_graph_embedding[:,0,:])

            # private_image_features = image_embeds[:,0,:] / image_embeds[:,0,:].norm(dim=1, keepdim=True)
            # private_text_features = output.hidden_states[7][:,0,:] / output.hidden_states[7][:,0,:].norm(dim=1, keepdim=True)

            # # cosine similarity as logits
            # logit_scale = self.logit_scale.exp()
            # # print(logit_scale)
            # batch_size = image_embeds.shape[0]

            # private_logits_per_image = logit_scale * private_image_features @ private_text_features.t()
            # private_logits_per_text = private_logits_per_image.t() # shape = [global_batch_size, global_batch_size]
            # probs = private_logits_per_image.softmax(dim=-1)
            # print(probs)
            # # cos = 0.
            # print("=====================")
            # for i in range(batch_size):
            #     print(private_logits_per_image[i, i])
                # cos += -private_logits_per_image[i, i]
            # cos_avg = cos / batch_size
            # print(cos_avg)
            


            # cos = 0.
            # for i in batch_size:
            #     private_logits_per_image[i, i]
            # print(private_logits_per_image.shape)
            # print(private_logits_per_image)
            # print("==========================")
            # print(private_logits_per_text)
            # private_labels = torch.arange(batch_size, dtype=torch.long).to(image.device)
            # private_loss = (F.cross_entropy(private_logits_per_image, private_labels) + F.cross_entropy(private_logits_per_text, private_labels)) / 2
            # print(private_loss)
            

            # loss = sum(1 / (2 * torch.exp(0.5)) * train_loss[i] + 0.5 / 2 for i in range(3))


            # bak =  torch.cat((share_graph_embedding[:,0,:], private_graph_embedding[:,0,:]), dim=1)
            # all_loss_private = self.compute_kl_loss(torch.cat((share_graph_embedding[:,0,:], private_graph_embedding[:,0,:]), dim=1), bak)
            graph_loss = self.compute_kl_loss(share_graph_embedding[:,0,:], private_graph_embedding[:,0,:])

            # text_loss = self.compute_kl_loss(output.hidden_states[7][:,0,:], output.last_hidden_state[:,0,:]) 
            # image_loss = self.compute_kl_loss(image_embeds[:,0,:], output.last_hidden_state[:,0,:]) 


            # share_graph_loss = self.compute_kl_loss(share_graph_embedding[:,0,:], self.proj_cls(output.last_hidden_state[:,0,:]))
            # private_graph_loss = self.compute_kl_loss(private_graph_embedding[:,0,:], self.proj_cls(output.last_hidden_state[:,0,:]))
            # graph_loss = share_graph_loss + private_graph_loss


            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image) 
                    output_m = self.text_encoder_m(text.input_ids, 
                                               attention_mask = text.attention_mask, 
                                               encoder_hidden_states = image_embeds_m,
                                               encoder_attention_mask = image_atts,        
                                               return_dict = True
                                              )           
                    prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   

                loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                    F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()
            else:
                #print(output.last_hidden_state[:,0,:].size(), graph_embedding.size())##[16:1:128]
                pre = self.proj(torch.cat((output.last_hidden_state[:,0,:],  private_graph_embedding[:,0,:], share_graph_embedding[:,0,:]), dim=1))
                # pre = torch.cat((self.proj(output.last_hidden_state[:,0,:]), graph_embedding[:,0,:]), dim=1)  #256
                
                prediction = self.cls_head(pre)
                loss = F.cross_entropy(prediction, targets)                
            return loss, utt_private_t[:,0,:], utt_private_v[:,0,:], utt_shared_t[:,0,:], utt_shared_v[:,0,:], share_loss + graph_loss
            
        else:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )        


            utt_private_t, utt_private_v, utt_shared_t, utt_shared_v = self.shared_private(output.hidden_states[7], image_embeds) 
            private_graph_embedding = self.private_graph(utt_private_v, utt_private_t)
            share_graph_embedding, _ = self.share_graph(utt_shared_v, utt_shared_t)

            # bt = image_embeds.size()[0]
            # print(bt)
            # weight = self.cos(share_graph_embedding[:,0,:], private_graph_embedding[:,0,:])

            # private_graph_embedding = self.private_graph(utt_private_v, utt_private_t)
            # share_graph_embedding = self.share_graph(utt_shared_v, utt_shared_t)
            pre = self.proj(torch.cat((output.last_hidden_state[:,0,:], private_graph_embedding[:,0,:], share_graph_embedding[:,0,:]), dim=1))
            # pre = self.proj(torch.cat((output.last_hidden_state[:,0,:], private_graph_embedding[:,0,:], share_graph_embedding[:,0,:]), dim=1))
            prediction = self.cls_head(pre)
            # prediction = self.cls_head(output.last_hidden_state[:,0,:])                        
            return prediction
 


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
                

