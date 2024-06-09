from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from Hateful_graph_final.graph_proj import Classifier_proj
from Hateful_graph_final.graph_classifier  import Classifier
import torch
from torch import nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn


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
        # self.cls_head = nn.Sequential(
        #           nn.Linear(128, 128),
        #           nn.ReLU(),
        #           nn.Linear(128, 2)#ve是三分类需要改成二分类
        #         )
        
        self.classifier_proj = Classifier_proj(768, 768)
        self.gat = dglnn.GATConv(768, 256, num_heads=1, feat_drop=0.4, activation=F.elu)
        self.classifier = Classifier(256, 128)
        self.proj = nn.Linear(896, 768)
        # self.proj = nn.Linear(768, 128)

        self.soft_mul = nn.Linear(896, 768)
        self.soft_graph = nn.Linear(896, 128)


        # self.hard_mul = nn.Linear(896, 768)
        # self.hard_graph = nn.Linear(896, 768)

        nn.init.kaiming_normal_(self.soft_mul.weight.data)
        nn.init.kaiming_normal_(self.soft_graph.weight.data)


        # self.map_individual_to_bi = nn.Linear(config.hidden_size, config.bi_hidden_size)  
        # self.map_bi_to_individual = nn.Linear(config.bi_hidden_size, config.hidden_size)  

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

    def fusion_soft(self, mul, graph):
        # mul = torch.relu(mul)
        # graph = torch.relu(graph)
        fusion_embeding = torch.cat((mul, graph), dim=1)
        soft_mul_embedding = torch.sigmoid(self.soft_mul(fusion_embeding))
        soft_graph_embedding = torch.sigmoid(self.soft_graph(fusion_embeding))
        pre = self.proj(
                torch.cat((soft_mul_embedding * mul, soft_graph_embedding * graph), dim=1))
        return pre

    def fusion_soft_768(self, mul, graph):
        # mul = torch.relu(mul)
        # graph = torch.relu(graph)
        fusion_embeding = torch.cat((mul, graph), dim=1)
        soft_mul_embedding = torch.sigmoid(self.soft_mul(fusion_embeding))
        soft_graph_embedding = torch.sigmoid(self.soft_graph(fusion_embeding))
        pre = self.proj(
                torch.cat((soft_mul_embedding * mul, soft_graph_embedding * graph), dim=1))
        return pre


    # def pre_sampling_sequence_soft(self, individual_sequence=None, sequence_c1=None, sequence_c2=None,
    #                                 modality=None):  # modality = v,t,pv
    #     individual_sequence = F.relu(individual_sequence)
    #     sequence_c1 = F.relu(sequence_c1)
    #     sequence_c2 = F.relu(sequence_c2)  # 16,36,1024
    #     feature_list = (individual_sequence, sequence_c1, sequence_c2)
    #     if modality == 'v':
    #         alpha_s = F.sigmoid(self.score_self_v(torch.cat(feature_list, 2)))
    #         alpha_c1 = F.sigmoid(self.score_cross1_v(torch.cat(feature_list, 2)))
    #         alpha_c2 = F.sigmoid(self.score_cross2_v(torch.cat(feature_list, 2)))
    #         # print(alpha_s.size())
    #         # print(individual_sequence.size())
    #         sequence_output = self.soft_v(
    #             torch.cat((individual_sequence * alpha_s, sequence_c1 * alpha_c1, sequence_c2 * alpha_c2), 2))

    #     elif modality == 't':
    #         alpha_s = F.sigmoid(self.score_self_t(torch.cat(feature_list, 2)))
    #         alpha_c1 = F.sigmoid(self.score_cross1_t(torch.cat(feature_list, 2)))
    #         alpha_c2 = F.sigmoid(self.score_cross2_t(torch.cat(feature_list, 2)))
    #         sequence_output = self.soft_t(
    #             torch.cat((individual_sequence * alpha_s, sequence_c1 * alpha_c1, sequence_c2 * alpha_c2), 2))

    #     elif modality == 'pv':
    #         alpha_s = F.sigmoid(self.score_self_pv(torch.cat(feature_list, 2)))
    #         alpha_c1 = F.sigmoid(self.score_cross1_pv(torch.cat(feature_list, 2)))
    #         alpha_c2 = F.sigmoid(self.score_cross2_pv(torch.cat(feature_list, 2)))
    #         sequence_output = self.soft_pv(
    #             torch.cat((individual_sequence * alpha_s, sequence_c1 * alpha_c1, sequence_c2 * alpha_c2), 2))

    #     return sequence_output

    # def pre_sampling_sequence(self, individual_sequence=None, sequence_c1=None, sequence_c2=None,
    #                             modality=None):  # modality = v,t,pv
    #     individual_sequence = F.relu(individual_sequence)
    #     sequence_c1 = F.relu(sequence_c1)
    #     sequence_c2 = F.relu(sequence_c2)  # 16,36,1024
    #     feature_list = (individual_sequence, sequence_c1, sequence_c2)
    #     if modality == 'v':
    #         alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_v(torch.cat(feature_list, 2))), dim=2)
    #         alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_v(torch.cat(feature_list, 2))), dim=2)
    #         alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_v(torch.cat(feature_list, 2))), dim=2)
    #     elif modality == 't':
    #         alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_t(torch.cat(feature_list, 2))), dim=2)
    #         alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_t(torch.cat(feature_list, 2))), dim=2)
    #         alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_t(torch.cat(feature_list, 2))), dim=2)
    #     elif modality == 'pv':
    #         alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_pv(torch.cat(feature_list, 2))), dim=2)
    #         alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_pv(torch.cat(feature_list, 2))), dim=2)
    #         alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_pv(torch.cat(feature_list, 2))), dim=2)

    #     ak = torch.cat((alpha_s, alpha_c1, alpha_c2), 2)  # 
    #     a_index = F.gumbel_softmax(ak, hard=True, dim=2)  #
    #     sequence_output = individual_sequence * (a_index[:, :, 0, :].squeeze(dim=2)) + sequence_c1 * (
    #         a_index[:, :, 1, :].squeeze(dim=2)) + sequence_c2 * (a_index[:, :, 2, :].squeeze(dim=2))
    #     return sequence_output


    # def get_sequence_pooled_output_final(self,
    #                                         sequence_output_t, sequence_output_v,
    #                                         all_attention_mask,
    #                                         sequence_output_pv_with_v, sequence_output_v_with_pv,
    #                                         all_attention_mask_v_pv,
    #                                         sequence_output_t_with_pv, sequence_output_pv_with_t,
    #                                         all_attention_mask_t_pv,
    #                                         individual_txt, individual_pv, individual_v):
    #     if self.if_pre_sampling == 1:  # hard 
    #         sequence_output_v = self.pre_sampling_sequence(individual_v, sequence_output_v, sequence_output_v_with_pv,
    #                                                         modality='v')  # 1024
    #         sequence_output_t = self.pre_sampling_sequence(individual_txt, sequence_output_t, sequence_output_t_with_pv,
    #                                                         modality='t')  # 768
    #         sequence_output_pv = self.pre_sampling_sequence(individual_pv, sequence_output_pv_with_v,
    #                                                         sequence_output_pv_with_t, modality='pv')  # 768

    #     elif self.if_pre_sampling == 0:  # mean
    #         sequence_output_v = (individual_v + sequence_output_v + sequence_output_v_with_pv) / 3
    #         sequence_output_t = (individual_txt + sequence_output_t + sequence_output_t_with_pv) / 3
    #         sequence_output_pv = (individual_pv + sequence_output_pv_with_v + sequence_output_pv_with_t) / 3

    #     elif self.if_pre_sampling == 2:  # soft
    #         v = F.sigmoid()


    #         sequence_output_v = self.pre_sampling_sequence_soft(individual_v, sequence_output_v,
    #                                                             sequence_output_v_with_pv, modality='v')  # 1024
    #         sequence_output_t = self.pre_sampling_sequence_soft(individual_txt, sequence_output_t,
    #                                                             sequence_output_t_with_pv, modality='t')  # 768
    #         sequence_output_pv = self.pre_sampling_sequence_soft(individual_pv, sequence_output_pv_with_v,
    #                                                                 sequence_output_pv_with_t, modality='pv')  # 768

    #     elif self.if_pre_sampling == 3:  # concat
    #         sequence_output_v = (sequence_output_v + sequence_output_v_with_pv) / 2
    #         sequence_output_t = (sequence_output_t + sequence_output_t_with_pv) / 2
    #         sequence_output_pv = (sequence_output_pv_with_v + sequence_output_pv_with_t) / 2

    #     pooled_output_v = self.map_bi_to_individual(torch.mean(sequence_output_v[:, 1:, :], dim=1))  # 1024-768
    #     pooled_output_t = torch.mean(sequence_output_t[:, 1:, :], dim=1)  # 768
    #     pooled_output_pv = torch.mean(sequence_output_pv[:, 1:, :], dim=1)  # 768

    #     return sequence_output_v, sequence_output_t, sequence_output_pv, pooled_output_v, pooled_output_t, pooled_output_pv
            
            
            
    def forward(self, image, text, targets, alpha=0, train=True):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        if train:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )  
            #prediction = self.cls_head(output.last_hidden_state[:,0,:])   

            ###########建图
            batch = image_embeds.size()[0]    
            # image_node = image_embeds.size()[1]
            image_node = image_embeds.size()[1] - 1
            text_node = output.hidden_states[6].size()[1] - 1
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
                g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(image.device)
                # g1.ndata['x'] = image_embeds[i,:,:]
                g1.ndata['x'] = image_embeds[i,1:,:]

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
                g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(image.device)
                g1.ndata['x'] = output.last_hidden_state[i,1:,:]
                g_t.append(g1)
            bg_t = dgl.batch(g_t)  ####批量文本图

            feat_i = bg_i.ndata['x']
            feat_t = bg_t.ndata['x']
            proj_i = self.classifier_proj(bg_i, feat_i)
            proj_t = self.classifier_proj(bg_t, feat_t)

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
                g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(image.device)
                g_f.append(g1)
            bg_f = dgl.batch(g_f)  ####批量融合图
            bg_f.ndata['x'] = torch.cat((proj_i, proj_t), dim=0)

            feat_f = bg_f.ndata['x']
            gat_f = self.gat(bg_f, feat_f)
            graph_embedding = self.classifier(bg_f, gat_f)  ###[B,128]

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

                # pre = self.proj(torch.cat((output.last_hidden_state[:,0,:], graph_embedding[:,0,:]), dim=1))


                # pre = torch.cat((self.proj(output.last_hidden_state[:,0,:]), graph_embedding[:,0,:]), dim=1)  #256
                pre = self.fusion_soft(output.last_hidden_state[:,0,:], graph_embedding[:,0,:])
                prediction = self.cls_head(pre)
                loss = F.cross_entropy(prediction, targets)                
            return loss 
            
        else:
            output = self.text_encoder(text.input_ids, 
                                       attention_mask = text.attention_mask, 
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,        
                                       return_dict = True
                                      )        

            batch = image_embeds.size()[0]    
            # image_node = image_embeds.size()[1]
            image_node = image_embeds.size()[1] - 1
            text_node = output.hidden_states[6].size()[1] - 1
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
                g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(image.device)
                # g1.ndata['x'] = image_embeds[i,:,:]
                g1.ndata['x'] = image_embeds[i,1:,:]
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
                g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(image.device)
                g1.ndata['x'] = output.last_hidden_state[i,1:,:]
                g_t.append(g1)
            bg_t = dgl.batch(g_t)  ####批量文本图

            feat_i = bg_i.ndata['x']
            feat_t = bg_t.ndata['x']
            proj_i = self.classifier_proj(bg_i, feat_i)
            proj_t = self.classifier_proj(bg_t, feat_t)

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
                g1 = dgl.graph((torch.tensor(u_i),torch.tensor(v_i))).to(image.device)
                g_f.append(g1)
            bg_f = dgl.batch(g_f)  ####批量文本图
            bg_f.ndata['x'] = torch.cat((proj_i, proj_t), dim=0)

            feat_f = bg_f.ndata['x']
            gat_f = self.gat(bg_f, feat_f)  ###tensor
            graph_embedding = self.classifier(bg_f, gat_f)  ###[B,128]

            # pre = self.proj(torch.cat((output.last_hidden_state[:,0,:], graph_embedding[:,0,:]), dim=1))
            pre = self.fusion_soft(output.last_hidden_state[:,0,:], graph_embedding[:,0,:])


            # pre = torch.cat((self.proj(output.last_hidden_state[:,0,:]), graph_embedding[:,0,:]), dim=1)  #256
            prediction = self.cls_head(pre)
            #prediction = self.cls_head(output.last_hidden_state[:,0,:])                        
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
                

