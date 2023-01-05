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

class LRBDataset(td.Dataset):
    def __init__(self, sents,labels,terms,tokenizer,args):
        self.sents = sents
        self.terms = terms
        self.tk = tokenizer
        self.max_length = args.max_length
        chunk_size = args.chunk_size
        nchunks = (len(sents)-1)// chunk_size + 1
        self.input_ids = []
        self.attention_mask = []
        for chunk_id in tqdm(range(nchunks),ascii=True,desc = 'Tokenizing dataset....'):
            start = chunk_id * chunk_size
            end = (chunk_id+1) * chunk_size
            chunk_tokenized_inputs = tokenizer(
                    sents[start:end],
                    padding='max_length',
                    truncation=True,
                    max_length=args.max_length,
                    is_split_into_words=True,
                    add_special_tokens=True
                )
            self.input_ids.extend(chunk_tokenized_inputs['input_ids'])
            self.attention_mask.extend(chunk_tokenized_inputs['attention_mask'])
        assert len(self.input_ids) == len(self.attention_mask) == len(self.sents)

        if args.lrb_num==2:
            if args.lrb_type == 'fewneg':
                self.labels = self.get_double_labels(labels)
            elif args.lrb_type == 'allneg':
                self.labels = self.get_all_labels(labels)
        elif args.lrb_num==1:
            self.labels = self.get_single_labels(labels)
    
    def get_single_labels(self,labels):
        new_labels = []
        for label in labels:
            label = list(label[:self.max_length-2])
            left_labels = [-100]*self.max_length
            right_labels = [-100]*self.max_length
            # right_labels = [-100] + label + [-100]
            # left_labels += [-100]*(self.max_length-len(left_labels))
            # right_labels += [-100]*(self.max_length-len(right_labels))

            for i in range(len(label)):
                if label[i]=='B':
                    left_labels[i+1] = 1
                    if i>1:
                        left_labels[i] = 0
                elif label[i]=='E':
                    right_labels[i+1] = 1
                    if i<self.max_length-3:
                        right_labels[i+2] = 0
                elif label[i]=='S':
                    left_labels[i+1] = 1
                    right_labels[i+1] = 1
                    if i>1:
                        left_labels[i] = 0
                    if i<self.max_length-3:
                        right_labels[i+2] = 0
            new_labels.append(left_labels[:]+right_labels[:])
        return new_labels

    def get_all_labels(self,labels):
        new_labels = []
        for label in labels:
            label = list(label[:self.max_length-2])
            left_labels = [-100]*(self.max_length-1)
            right_labels = [-100]*(self.max_length-1)
            for i in range(len(label)):
                if label[i]=='B':
                    left_labels[i] = 1
                elif label[i]=='E':
                    right_labels[i+1] = 1
                elif label[i]=='S':
                    left_labels[i] = 1
                    right_labels[i+1] = 1
                else:
                    left_labels[i] = 0
                    right_labels[i+1] = 0
            new_labels.append(left_labels[:]+right_labels[:])
        return new_labels
        
    def get_double_labels(self,labels):
        new_labels = []
        for label in labels:
            label = list(label[:self.max_length-2])
            left_labels = [-100]*(self.max_length-1)
            right_labels = [-100]*(self.max_length-1)
            for i in range(len(label)):
                if label[i]=='B':
                    left_labels[i] = 1
                    if i>1:
                        left_labels[i-1] = 0
                elif label[i]=='E':
                    right_labels[i+1] = 1
                    if i<self.max_length-3:
                        right_labels[i+2] = 0
                elif label[i]=='S':
                    left_labels[i] = 1
                    right_labels[i+1] = 1
                    if i>1:
                        left_labels[i-1] = 0
                    if i<self.max_length-3:
                        right_labels[i+2] = 0
            new_labels.append(left_labels[:]+right_labels[:])
        return new_labels


    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        sent = self.sents[idx]
        label = torch.LongTensor(self.labels[idx])
        input_ids = torch.LongTensor(self.input_ids[idx])
        attention_mask = torch.LongTensor(self.attention_mask[idx])
        return input_ids,attention_mask,label,sent

    def collate_fn(self, batch):
        input_ids, attention_mask, labels, sents = zip(*batch)
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        return input_ids, attention_mask, labels, sents