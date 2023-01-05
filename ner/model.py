import torch
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF

class NERModel(BertPreTrainedModel):
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, pad_mask = None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # crf with mask
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if self.eval():
            # apply softmax to logits
            logits = torch.softmax(logits, dim=-1)

        
        # if labels is not None:
        #     loss = -self.crf(logits, labels, mask=pad_mask)
            # return loss
        # else:
        #     # return self.crf.decode(logits, mask=pad_mask)
        #     return logits
        return logits,loss


class NERThreeModel(BertPreTrainedModel):
    def __init__(self, config):
        super(NERThreeModel, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        self.three_idx = torch.LongTensor([(i,i+1,i+2) for i in range(0,config.max_length-2)])
        # print shape of threeidx
        print('three_idx shape:',self.three_idx.shape)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, pad_mask = None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # bs,seq_len,hs
        triple_indexes = sequence_output[:,self.three_idx,:]
        triple_indexes = triple_indexes.flatten(start_dim = 2, end_dim = 3)
        # print(self.classifier.in_features)
        # print(self.classifier.out_features)
        # print(triple_indexes.shape)
        # from IPython import embed; embed()
        logits = self.classifier(triple_indexes)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if self.eval():
            logits = torch.softmax(logits, dim=-1)
        return logits,loss