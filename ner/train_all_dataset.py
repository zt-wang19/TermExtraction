import torch.utils.data as td
import json
import torch
from tqdm import tqdm
from utils import *

def read_from_file(path):
    # get sents,labels from txt
    sents = []
    labels = []
    with open(path,'r') as f:
        for i,line in enumerate(tqdm(f)):
            if i%2==0:
                sents.append(list(line.strip()))
            else:
                labels.append(line.strip())
    return sents,labels

def read_terms(path):
    # get terms from lines
    terms = []
    with open(path,'r') as f:
        for line in f:
            terms.append(line.strip())
    return terms


class NERTrainDataset(td.Dataset):
    def __init__(self, sents,labels,terms,tokenizer,args):
        self.all_data = [[a,b] for a,b in zip(sents,labels)]
        self.terms = terms
        self.tk = tokenizer
        self.max_length = args.max_length
        self.args = args

        self.data_split = args.data_split
        self.data_len = len(sents)

        x = (len(sents)//self.data_split)+1
        self.split_idx = [(x*i,x*(i+1))  for i in range(self.data_split)]
        self.update_split(0)

    def update_split(self, split):
        assert split>=0 and split<self.data_split
        split_data = self.all_data[self.split_idx[split][0]:self.split_idx[split][1]]
        sents = [list(data[0]) for data in split_data]
        labels = [data[1] for data in split_data]

        self.sents = sents
        self.labels = labels

        chunk_size = 16
        chunked_sents = [sents[i:i+chunk_size] for i in range(0, len(sents), chunk_size)]
        self.input_ids = []
        self.attention_mask = []
        for chunk in tqdm(chunked_sents, desc=f'tokenizing split{split}'):
            chunk_tokenized_inputs =self.tk(
                    chunk,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    is_split_into_words=True,
                )
            self.input_ids.extend(chunk_tokenized_inputs['input_ids'])
            self.attention_mask.extend(chunk_tokenized_inputs['attention_mask'])
        
        assert len(self.input_ids) == len(self.attention_mask) == len(self.sents)

    
    def __len__(self):
        return len(self.sents)
    def __getitem__(self, idx):

        sent = self.sents[idx]
        label = [label2id[x]for x in self.labels[idx]]
        if self.args.use3:
            label = self.pad3label(label)
        else:
            label = self.pad_truncate_label(label)
        label = torch.LongTensor(label)
        input_ids = torch.LongTensor(self.input_ids[idx])
        attention_mask = torch.LongTensor(self.attention_mask[idx])
        pad_mask = torch.where(label==-100,torch.tensor(0),torch.tensor(1)).bool()
        return input_ids,attention_mask,torch.LongTensor(label),pad_mask,sent

    def pad3label(self, label):
        label = label[:self.max_length-2]
        label = label + [-100]*(self.max_length-2-len(label))
        return label
    
    def pad_truncate_label(self, label):
        
        label = [-100]+label[:self.max_length-2]+[-100]
        label = label + [-100]*(self.max_length-len(label))
        return label

    # def pad_truncate_no_special(self,label):
    #     if len(label) > self.max_length:
    #         label = label[:self.max_length]
    #     else:
    #         label = label + [0]*(self.max_length-len(label))
    #     return label

    def ner_collate_fn(self,batch):
        input_ids, attention_mask, labels, pad_mask, sents = zip(*batch)
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        pad_mask = torch.stack(pad_mask, dim=0)
        return input_ids, attention_mask, labels, pad_mask, sents