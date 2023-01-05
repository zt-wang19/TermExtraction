import torch
from transformers import BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class LRBModel(BertPreTrainedModel):

    def __init__(self, config):
        super(LRBModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.max_length = config.max_length

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.left_clsf = nn.Linear(config.hidden_size*2, config.num_labels)
        self.right_clsf = nn.Linear(config.hidden_size*2, config.num_labels)
        # indexes like [(0,1),(1,2),...,(n-1,n)]
        self.double_idxes = torch.LongTensor([(i,i+1) for i in range(0,self.max_length-1)])
        print('double_idx.shape:,',self.double_idxes.shape)

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
        double_indexes = sequence_output[:,self.double_idxes,:].flatten(start_dim = 2, end_dim = 3)
        left_logits = self.left_clsf(double_indexes)
        right_logits = self.right_clsf(double_indexes)
        logits = torch.cat([left_logits,right_logits],dim = 1)
        loss = None
        loss_func = nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
        return logits,loss

class LRBOneModel(BertPreTrainedModel):

    def __init__(self, config):
        super(LRBOneModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.max_length = config.max_length

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.left_clsf = nn.Linear(config.hidden_size, config.num_labels)
        self.right_clsf = nn.Linear(config.hidden_size, config.num_labels)
        # indexes like [(0,1),(1,2),...,(n-1,n)]
        # self.double_idxes = torch.LongTensor([(i,i+1) for i in range(0,self.max_length-1)])
        # print('double_idx.shape:,',self.double_idxes.shape)

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
        # double_indexes = sequence_output[:,self.double_idxes,:].flatten(start_dim = 2, end_dim = 3)
        left_logits = self.left_clsf(sequence_output)
        right_logits = self.right_clsf(sequence_output)
        logits = torch.cat([left_logits,right_logits],dim = 1)
        loss = None
        loss_func = nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
        return logits,loss