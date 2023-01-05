from turtle import left
import torch
import torch.nn as nn
import torch
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy.random as random
import math
from sklearn.metrics import precision_score, recall_score, f1_score,precision_recall_curve
import numpy as np
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# import pandas as pd

def average_precision_overal_recall_interval(final_logits_pos, y_true,  rec_low=0.9, rec_high=0.98):
    prec, rec, thres = precision_recall_curve(y_true, final_logits_pos)
    rec_interest = np.where((rec>rec_low) *(rec<rec_high))[0]
    rec_i = rec[rec_interest]
    prec_i = prec[rec_interest]
    ave_prec = np.sum(np.diff(rec_i) * np.array(prec_i)[:-1])/np.sum(np.diff(rec_i))
    return ave_prec


class TermModel(BertPreTrainedModel):
    def __init__(self, config, cneg=False):
        super(TermModel, self).__init__(config)
        # self.config.args = config.args
        self.num_labels = config.num_labels
        self.config = config
        self.cneg = cneg


        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        self.inner_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.leftright_key_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.leftright_value_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.term_key_transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.term_value_transform = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()
        self.step_id = 0
    



    def compute_dynamic_stat(self, step_id):
        self.dynamic_state = {}
        if self.config.args.neg_ratio_func is not None:
            self.dynamic_state['neg_ratio'] = self.config.args.neg_ratio_func(step_id)
        else:
            self.dynamic_state['neg_ratio'] = self.config.args.neg_ratio
        
        if self.config.args.sample_probs_func is not None:
            self.dynamic_state['sample_probs'] = self.config.args.sample_probs_func(step_id)
        else:
            self.dynamic_state['sample_probs'] = self.config.args.sample_probs
        # if step_id % 500 == 0:
        #     print("dynamic state", self.dynamic_state)

        
        # self.dynamic_state['context_size'] = self.config.args.context_size

    def forward(self, input_ids, input_sents, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, real_lens = None,cnegs = None
                ):
        # from IPython import embed; embed(header="forward")
        
        self.compute_dynamic_stat(self.step_id)
        self.step_id += 1

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)
        
        self.max_term_size = max([pair[1]-pair[0]+1 for x in labels for pair in x if len(x) > 0])
        flatten_labels, in_batch_ids, loss_labels, termlabels, leftcontextlabels, rightcontextlabels = self.get_negative(input_sents, labels, real_lens, max_term_size=self.max_term_size, context_size=self.config.args.context_size, neg_ratio=self.dynamic_state['neg_ratio'],cnegs=cnegs)


        # from IPython import embed; embed(header="forward")


        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        final_vec, final_vec2 = self.get_hiddens(sequence_output, in_batch_ids, termlabels, leftcontextlabels, rightcontextlabels, max_term_size=self.max_term_size, context_size=self.config.args.context_size)

        if self.config.args.use_two_classifier == 1:
            final_logits = (self.classifier(final_vec) + self.classifier2(final_vec2))/2
        else:
            final_logits = self.classifier(final_vec)


        # from IPython import embed; embed(header="final_logits")
        

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(final_logits, loss_labels.to(final_logits.device))
        return final_logits,loss

    def inference(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, real_lens = None, 
                debug = False 
                ):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)
        max_term_size = self.max_term_size = 15

        # self.get_ngrams(input_ids, real_lens, max_term_size=max_term_size, context_size=3)
        # self.max_term_size = max([pair[1]-pair[0]+1 for x in labels for pair in x if len(x) > 0])
        flatten_labels, in_batch_ids, loss_labels, termlabels, leftcontextlabels, rightcontextlabels = self.get_ngrams(labels, real_lens, max_term_size=max_term_size, context_size=self.config.args.context_size, sample=False)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        final_vec, final_vec2 = self.get_hiddens(sequence_output, in_batch_ids, termlabels, leftcontextlabels, rightcontextlabels, max_term_size=self.max_term_size, context_size=self.config.args.context_size)
        if self.config.args.use_two_classifier == 1:
            final_logits = (self.classifier(final_vec) + self.classifier2(final_vec2))/2
        else:
            final_logits = self.classifier(final_vec)

        ret = (final_logits.detach().cpu().numpy(),loss_labels.cpu().numpy())
        if debug:
            ret = (ret[0], ret[1],  in_batch_ids, termlabels)

        return ret
        

    def prediction(self, final_logits, loss_labels):
        # final_logits = final_logits.detach().cpu().numpy()
        predictions = final_logits.argmax(axis=1)
        # loss_labels = loss_labels.cpu().numpy()
        f1 =  f1_score(loss_labels, predictions)
        precision = precision_score(loss_labels, predictions)
        recall = recall_score(loss_labels, predictions)

        false_negative_id = np.where(((1 - predictions) * loss_labels) == 1)
        false_positive_id = np.where((predictions * (1-loss_labels)) == 1)

        # from IPython import embed; embed(header="precision recall curve")
        ave_prec = average_precision_overal_recall_interval(final_logits_pos=final_logits[:,1], y_true=loss_labels, rec_low=0.90, rec_high=0.97)

        return f1, precision, recall, false_negative_id, false_positive_id, ave_prec
    

    
    def inference_one_sent(self, sentence, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, real_lens = None,  top_k=30, verbose=True,sm=False,smt = 1
                ):
        encoded_sent = self.tk(sentence, return_tensors='pt', max_length=512, is_split_into_words=True,truncation=True)
        # from IPython import embed; embed(header="inference_one_sent")

        outputs = self.bert(encoded_sent['input_ids'].to(self.device),
                    attention_mask=encoded_sent['attention_mask'].to(self.device),
                    token_type_ids=encoded_sent['token_type_ids'].to(self.device),
                    position_ids=position_ids, 
                    head_mask=head_mask)
        max_term_size = self.max_term_size = min( len(encoded_sent['input_ids'][0]) -2, 15)
        flatten_labels, in_batch_ids, loss_labels, termlabels, leftcontextlabels, rightcontextlabels = self.get_ngrams(labels=[[]], real_lens=[len(encoded_sent['input_ids'][0])], max_term_size=max_term_size, context_size=self.config.args.context_size, sample=False)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        final_vec, final_vec2 = self.get_hiddens(sequence_output, in_batch_ids, termlabels, leftcontextlabels, rightcontextlabels, max_term_size=self.max_term_size, context_size=self.config.args.context_size)
        if self.config.args.use_two_classifier == 1:
            final_logits = (self.classifier(final_vec) + self.classifier2(final_vec2))/2
        else:
            final_logits = self.classifier(final_vec)

        # from IPython import embed; e/mbed(header="inference_one_sent")
        # print(final_logits)
        out = self.view_inference_result(final_logits, flatten_labels, self.tk.convert_ids_to_tokens(encoded_sent['input_ids'][0]), top_k=top_k, verbose=verbose,sm=sm,smt=smt)
        return out
    
    def view_inference_result(self, final_logits, flatten_labels, input_tokens, top_k=30, verbose=True,sm = False,smt=1):
        if sm:
            final_logits = torch.softmax(final_logits/smt,dim=-1)
        final_logits = final_logits.detach().cpu().numpy()
        prediction_1 = final_logits[:, 1]
        # sort by prediction
        sorted_prediction_idx = np.argsort(-prediction_1) # from large to small
        # sort flatten_labels according to sorted_prediction
        sorted_flatten_labels = [flatten_labels[i] for i in sorted_prediction_idx]

        sorted_prediction_1 = [prediction_1[i] for i in sorted_prediction_idx]
        # get top-k predictions
        # top_k = 30
        top_k_flatten_labels = sorted_flatten_labels[:top_k]
        top_k_prediction_1 = sorted_prediction_1[:top_k]
        
        # get the span in input_tokens marked by flatten_labels
        top_k_span = [(input_tokens[pair[0]: pair[1]+1],pair) for pair in top_k_flatten_labels]

        output = list(zip(top_k_span, top_k_prediction_1))
        if verbose:
            print(output)
        return output



    
    def get_ngrams(self, labels, real_lens, max_term_size, context_size, sample=False):
        '''
        get the ngrams for each term
        '''
        flatten_labels = []
        in_batch_ids = []
        loss_labels = []
        termlabels = []
        leftcontextlabels = []
        rightcontextlabels = []
        for in_batch_id, (label, real_len) in enumerate(zip(labels, real_lens)):
            for left in range(1, real_len-1):
                for right in range(left, real_len-1):
                    if right - left + 1 > max_term_size:
                        continue
                    if (left, right) in label:
                        loss_label = 1
                    else:
                        if sample:
                            if random.uniform(0,1) > 0.2:
                                continue
                        loss_label = 0
                    flatten_labels.append((left, right))
                    in_batch_ids.append(in_batch_id)
                    loss_labels.append(loss_label)

        termlabels = []
        for label in flatten_labels:
            tmp = [x for x in range(label[0], label[1]+1)] + [-1] * (max_term_size - (label[1]+1 - label[0]))
            if len(tmp) != max_term_size:
                from IPython import embed; embed(header='label error')
            termlabels.append(tmp)
        
        leftcontextlabels = []
        for label in flatten_labels:
            tmp = [x for x in range(max(label[0]-context_size,0), label[0])] + [-1] * (context_size - label[0])
            leftcontextlabels.append(tmp)
        
        rightcontextlabels = []
        for _idx, label in enumerate(flatten_labels):
            sent_len = real_lens[in_batch_ids[_idx]]
            tmp = [x for x in range(label[1]+1, min(label[1]+1+context_size, sent_len))] + [-1] * (context_size- (sent_len - label[1]-1))
            if len(tmp) != context_size:
                from IPython import embed; embed(header='right label error')
            # print(tmp)
            rightcontextlabels.append(tmp)
        
        flatten_labels = torch.LongTensor(flatten_labels)
        in_batch_ids = torch.LongTensor(in_batch_ids)
        loss_labels = torch.LongTensor(loss_labels)
        termlabels = torch.LongTensor(termlabels)
        leftcontextlabels = torch.LongTensor(leftcontextlabels)
        rightcontextlabels = torch.LongTensor(rightcontextlabels)
            
        return flatten_labels, in_batch_ids, loss_labels, termlabels, leftcontextlabels, rightcontextlabels

    
    def unitTest1(self, input_ids, flatten_labels, in_batch_ids, loss_labels, termlabels, leftcontextlabels, rightcontextlabels, view_sample_id=0):
        '''check the sampled data is correct'''
        view_id = torch.where(in_batch_ids == view_sample_id)
        tempsent = self.tk.convert_ids_to_tokens(input_ids[view_sample_id])
        for (x,y), loss_id, t_l, l_l, r_l  in zip(flatten_labels[view_id], loss_labels[view_id], termlabels[view_id], leftcontextlabels[view_id], rightcontextlabels[view_id]):
            print(tempsent[x: y+1], loss_id, t_l, l_l, r_l)
        


    
    # def flatten_labels(self, flatten_labels):
    #     for label in zip(flatten_labels):
    #         flatten_labels.append(label)
    #         loss_labels.append(1)
    #         for neg in neg_label:
    #             flatten_labels.append(neg)
    #             loss_labels.append(0)
    #     flatten_labels = torch.LongTensor(flatten_labels)
    #     loss_labels = torch.LongTensor(loss_labels)
    #     return flatten_labels, loss_labels

    # def get_mask(self, label, sent_len, max_term_size=15,  context_size=3, query_num=3):
    #     # left mask
    #     BIGNUM = 1000000
    #     left_mask = torch.zeros(context_size)
    #     if label[0] - context_size < 0:
    #         left_mask[:context_size - label[0]] = - BIGNUM
    #     # term_mask
    #     term_mask = torch.zeros(max_term_size)
    #     term_mask = term_mask[(label[1] - label[0])+1:] = - BIGNUM
    #     # right mask
    #     right_mask = torch.zeros(context_size)
    #     if label[1] + context_size > sent_len:
    #         right_mask[sent_len - label[1]:] = - BIGNUM
    #     return left_mask, term_mask, right_mask
    
    def get_hiddens(self, seq_output, in_batch_ids, termlabels, leftcontextlabels, rightcontextlabels, max_term_size=15, context_size=3):
        BIGNUM = 1000000.0

        batch_size = seq_output.size(0)

        expand_in_batch_ids_left = in_batch_ids.repeat((context_size,1)).transpose(0,1).reshape(-1)
        left_hidden_tensor = seq_output[expand_in_batch_ids_left, leftcontextlabels.reshape(-1)].reshape(in_batch_ids.shape[0], context_size,-1)

        expand_in_batch_ids_term = in_batch_ids.repeat((max_term_size,1)).transpose(0,1).reshape(-1)
        term_hidden_tensor = seq_output[expand_in_batch_ids_term, termlabels.reshape(-1)].reshape(in_batch_ids.shape[0], max_term_size,-1)
        term_hidden_tensor = term_hidden_tensor.reshape(in_batch_ids.shape[0], max_term_size, -1)

        expand_in_batch_ids_right = in_batch_ids.repeat((context_size,1)).transpose(0,1).reshape(-1)
        right_hidden_tensor = seq_output[expand_in_batch_ids_right, rightcontextlabels.reshape(-1)].reshape(in_batch_ids.shape[0], context_size,-1)

        # from IPython import embed; embed(header="in get_hiddens")
        # prepare key 
        term_hidden_tensor[torch.where(termlabels==-1)] = 0.0
        query_vector = term_hidden_tensor.mean(dim=1, keepdim=True)
        query_vector = self.inner_query(query_vector)

        # query vector 2

        # from IPython import embed; embed(header="get_hiddens")
        term_labels_left_bound = termlabels[:,0]
        term_labels_left_out_bound = term_labels_left_bound - 1
        term_labels_right_bound = torch.max(termlabels, dim=-1).values
        term_labels_right_out_bound = term_labels_right_bound + 1

        term_labels_left_bound_hiddens = seq_output[in_batch_ids, term_labels_left_bound]
        term_labels_left_out_bound_hiddens = seq_output[in_batch_ids, term_labels_left_out_bound]
        term_labels_right_out_bound_hiddens = seq_output[in_batch_ids, term_labels_right_out_bound]
        term_labels_right_bound_hiddens = seq_output[in_batch_ids, term_labels_right_bound]

        bound_out = term_labels_left_out_bound_hiddens + term_labels_right_out_bound_hiddens
        bound_in = term_labels_left_bound_hiddens + term_labels_right_bound_hiddens


        # margin_hidden_tensor[torch.where()]

        term_hidden_tensor_key =  self.term_key_transform(term_hidden_tensor)
        term_hidden_tensor_value = self.term_value_transform(term_hidden_tensor)
        term_attn = term_hidden_tensor_key @ query_vector.transpose(1,2)/math.sqrt(query_vector.size(-1))
        term_attn = term_attn.squeeze(2)
        term_attn[torch.where(termlabels==-1)] = - BIGNUM
        term_attn_score = torch.softmax(term_attn, dim=-1)
        term_final_vec = (term_attn_score.unsqueeze(-1) * term_hidden_tensor_value).sum(1)

        # print(query_vector.shape)
        # print(term_final_vec.shape)
        # from IPython import embed; embed()
        query_vector = term_final_vec.unsqueeze(1)


        # attention
        left_hidden_tensor_key =  self.leftright_key_transform(left_hidden_tensor)
        left_hidden_tensor_value = self.leftright_value_transform(left_hidden_tensor)
        left_attn = left_hidden_tensor_key @ query_vector.transpose(1,2)/math.sqrt(query_vector.size(-1))
        left_attn = left_attn.squeeze(2)
        left_attn[torch.where(leftcontextlabels==-1)] = - BIGNUM
        left_attn_score = torch.softmax(left_attn, dim=-1)
        left_final_vec = (left_attn_score.unsqueeze(-1) * left_hidden_tensor_value).sum(1)


        

        right_hidden_tensor_key =  self.leftright_key_transform(right_hidden_tensor)
        right_hidden_tensor_value = self.leftright_value_transform(right_hidden_tensor)
        right_attn = right_hidden_tensor_key @ query_vector.transpose(1,2)/math.sqrt(query_vector.size(-1))
        right_attn = right_attn.squeeze(2)
        right_attn[torch.where(rightcontextlabels==-1)] = - BIGNUM
        right_attn_score = torch.softmax(right_attn, dim=-1)
        right_final_vec = (right_attn_score.unsqueeze(-1) * right_hidden_tensor_value).sum(1)


        final_vec = torch.cat((left_final_vec, term_final_vec, right_final_vec), dim=-1)

        final_vec2 =  torch.cat([bound_out, bound_in], dim=-1)

        return final_vec, final_vec2

        
        








    def get_negative(self, input_sents, labels, sent_lens, neg_ratio=2, least_neg_num=2, max_term_size=15, context_size=3, resample=False,cnegs= None):
        flatten_labels = []
        in_batch_ids = []
        loss_labels = []
        for in_batch_id, (common_word, label, sent_len) in enumerate(zip(input_sents, labels, sent_lens)):
            if resample:
                pass
            else:
                if len(label) > 0:
                    flatten_labels.extend(label)
                    in_batch_ids.extend([in_batch_id] * len(label))
                    loss_labels.extend([1] * len(label))
            if self.cneg:
                cneg = cnegs[in_batch_id]


            for neg_id in range(max(neg_ratio*len(label), least_neg_num)):
                neg = self.sample(common_word, label, sent_len, probs=self.dynamic_state['sample_probs'],cneg = cneg,max_term_size=max_term_size)
                flatten_labels.append(neg)
                in_batch_ids.append(in_batch_id)
                loss_labels.append(0)

        termlabels = []
        for label,loss_label in zip(flatten_labels,loss_labels):
            tmp = [x for x in range(label[0], label[1]+1)] + [-1] * (max_term_size - (label[1]+1 - label[0]))
            if len(tmp) != max_term_size:
                from IPython import embed; embed(header='flat label error')
            termlabels.append(tmp)
        
        leftcontextlabels = []
        for label in flatten_labels:
            tmp = [x for x in range(max(label[0]-context_size,0), label[0])] + [-1] * (context_size - label[0])
            leftcontextlabels.append(tmp)
        
        rightcontextlabels = []
        for _idx, label in enumerate(flatten_labels):
            sent_len = sent_lens[in_batch_ids[_idx]]
            tmp = [x for x in range(label[1]+1, min(label[1]+1+context_size, sent_len))] + [-1] * (context_size- (sent_len - label[1]-1))
            if len(tmp) != context_size:
                from IPython import embed; embed(header='right label error')
            # print(tmp)
            rightcontextlabels.append(tmp)
        




         
        flatten_labels = torch.LongTensor(flatten_labels)
        in_batch_ids = torch.LongTensor(in_batch_ids)
        loss_labels = torch.LongTensor(loss_labels)
        termlabels = torch.LongTensor(termlabels)
        leftcontextlabels = torch.LongTensor(leftcontextlabels)
        rightcontextlabels = torch.LongTensor(rightcontextlabels)

        
        return flatten_labels, in_batch_ids, loss_labels, termlabels, leftcontextlabels, rightcontextlabels
    

    
    def sample(self, common_word, label, sent_len, probs={"random":0.5,"overlap":0.3,"concate":0.2},cneg=None,max_term_size = 10):
        s = random.uniform(0, 1)
        # from IPython import embed; embed(header="in sample")
        for key, p in probs.items():
            s -= p
            if s <= 0:
                if key == "random":
                    neg = self.random_sample(label, sent_len)
                    # print('random:',neg)
                elif key == "overlap":
                    neg = self.overlap_sample(label, sent_len)
                elif key == "concate":
                    neg = self.concate_sample(label, sent_len)
                elif key == "common":

                    neg = self.common_sample(common_word, label, sent_len,cneg,max_term_size)
                    # print('cneg:',neg)
                return neg
    
    def common_sample(self, common_word, label, sent_len,neg,max_term_size):
        total = len(neg)
        if total == 0:
            return self.random_sample(label, sent_len)

        ok = False
        count = 0
        res=None
        while not ok:
            count += 1
            if count > 3:
                return self.random_sample(label, sent_len)
            s = random.randint(0, total)
            # neg = common_word[s]
            res = neg[s]
            if neg not in label and res[1]-res[0]<max_term_size:
                # print('common ok!')
                ok = True
            # else:
            #     print('common not ok!')
        return res

    
        

    def random_sample(self, label, sent_len, window_max=10):
        window_max = min(window_max, self.max_term_size -1)
        ok = False
        cnt = 0
        while not ok:
            
            s = random.randint(1, sent_len-1)
            window = random.randint(1, window_max+1)
            t = s
            t = random.randint(max(1, s-window), min(sent_len-1, s+window+1))
            neg = (s, t) if s <= t else (t, s)
            if neg not in label or cnt > 10:
                ok = True
            cnt+=1
        return neg

    def overlap_sample(self, label, sent_len):
        label_num = len(label)
        if label_num == 0:
            return self.random_sample(label, sent_len) # no label, sample randomly
        
        labelid = random.randint(0, label_num)
        neg = label[labelid]
        count = 0 
        while neg in label:
            count += 1
            if count > 2:
                return self.random_sample(label, sent_len)
            labelid = random.randint(0, label_num)
            neg = label[labelid]
            random_float = random.uniform(0,1)
            if random_float<0.3: # add to both side
                left_move = random.randint(1, 2)
                right_move = random.randint(1, 2)
                s = max(1, neg[0] - left_move) # 1: not include the first one [CLS]
                t = min(sent_len - 2, neg[1] + right_move) # -2: not include the last one [SEP]
                neg = (s, t)
            elif random_float<0.65: # add to left
                left_move = random.randint(1, 2)
                s = max(1, neg[0] - left_move)
                t = neg[1]
                neg = (s, t)
            else: # add to right
                right_move = random.randint(1, 2)
                s = neg[0]
                t = min(sent_len - 2, neg[1] + right_move) # -2: not include the last one [SEP]
                neg = (s, t)
        if t - s + 1 > self.max_term_size:
            return self.random_sample(label, sent_len) # too long, sample randomly
        return neg
    
    def concate_sample(self, label, sent_len):
        label_num = len(label)
        if label_num <= 1:
            return self.random_sample(label, sent_len) # only one/none label, use random sample
        count = 0
        while True:
            count += 1
            if count > 2: # reduce to random sample
                return self.random_sample(label, sent_len)
            indices = random.choice(label_num, 2, replace=False)
            index_left, index_right= (indices[0], indices[1]) if indices[0]<indices[1] else (indices[1], indices[0])
            pos_left, pos_right = label[index_left], label[index_right]
            if pos_left[1] + 5 > pos_right[0] and (pos_left[0], pos_right[1]) not in label:
                break
        neg = (pos_left[0], pos_right[1])
        if neg[1] - neg[0] + 1 > self.max_term_size:
            return self.random_sample(label, sent_len) # too long, sample randomly
        return neg

        



        
