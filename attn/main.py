import os
# from code.get_args import get_parser
from get_args import get_parser
# import get_args
from model import TermModel
import json
import torch
import torch.nn as nn
from transformers import BertTokenizerFast
from dataset import *
from train import *
import torch.utils.data as td
from samp import *
from dataset import *

parser = get_parser()
args = parser.parse_args()



def static_sample1(step=0):
    return {"random":0.5,"overlap":0.3,"concate":0.2}
    # return {"random":0.5,"common":0.5}
def static_sample2(step):
    return {"random":0.7,"overlap":0.1,"concate":0.2}
def static_sample3(step):
    return {"random":0.3,"overlap":0.3,"concate":0.4}
def static_sample4(step):
    return {"random":0.4,"overlap":0.2,"concate":0.2, "common": 0.2}

def dynamic_sample1(step):
    if step < 15000:
        return static_sample2(step)
    else:
        return static_sample1(step)

# def dynamic_sample1(step):
#     return {"random":,"overlap":0.3,"concate":0.3}
# def dynamic_sample2(step):
#     return {"random":0.3,"overlap":0.3,"concate":0.3}

def neg_ratio_static1(step):
    return 2

def neg_ratio_static2(step):
    return 1

def neg_ratio_dynamic1(step):
    return min(max(1, int(3.5 - step/10000)), 3)

def neg_ratio_dynamic2(step):
    return min(max(1, int(0.5 + step/10000)), 3)

def neg_ratio_static3(step):
    return 4

def neg_ratio_static4(step):
    return 5

def neg_ratio_dynamic3(step):
    if step < 15000:
        return 5
    elif step <25000:
        return 1
    else:
        return 3

def neg_ratio_dynamic4(step):
    if step < 15000:
        return 5
    else:
        return 3



    

sample_probs_funcs = [wzt_sample1, wzt_sample2, wzt_sample3, wzt_sample1, wzt_sample4]
neg_ratio_funcs = [neg_ratio_static1, neg_ratio_static2, neg_ratio_dynamic1, neg_ratio_dynamic2, neg_ratio_static3, neg_ratio_static4, neg_ratio_dynamic3, neg_ratio_dynamic4]

setattr(args, 'neg_ratio_func', neg_ratio_funcs[args.neg_ratio_choice])
setattr(args, 'sample_probs_func', sample_probs_funcs[args.sample_probs_choice])
print(args)
print(args.sample_probs_func())

# args = parser.parse_args(['--train_path', '../huawei_dataset/pc.txt','--valid_path','../huawei_dataset/medicine.txt'])

tk = BertTokenizerFast.from_pretrained(args.model_name)


# valid_sents,valid_labels = get_sents_labels(args.valid_path,args.max_length)
# valid_pos_ngrams = get_ngrams(valid_sents,valid_labels,args.max_length)
# valid_terms = get_terms(valid_pos_ngrams)

train_data = []
with open(args.train_path, 'r', encoding='utf-8') as f:
    # train_sents = f.readlines()
    for line in tqdm(f,desc='read train data'):
        train_data.append(json.loads(line))

valid_data = []
with open(args.valid_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f,desc='read valid data'):
        valid_data.append(json.loads(line))

train_dataset = NegDataset(train_data, tk,args)
valid_dataset = NegDataset(valid_data, tk,args)

train_terms = []

def read_terms(path):
    # get terms from lines
    terms = []
    with open(path,'r') as f:
        for line in f:
            terms.append(line.strip())
    return terms
valid_terms = read_terms(args.valid_terms_path)

train_dataloader = td.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,collate_fn=NegDataset.collate_fn)
valid_dataloader = td.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,collate_fn=SophDatasetNegSampling.collate_fn)


# from IPython import embed; embed(header="in soph 48")
from torch.optim import AdamW
model = TermModel.from_pretrained(args.model_name, num_labels=2,cneg=True)
setattr(model.config, 'args', args)

initialize_from_existing_ckpt = not (args.finetuned_ckpt == "" or args.finetuned_ckpt is None)

if initialize_from_existing_ckpt:
    print("loading finetuned ckpt")
    model.load_state_dict(torch.load(args.finetuned_ckpt))
setattr(model, 'tk', tk)
# model = nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,
                    lr=args.lr,
                    eps=args.eps
                    )


train_model(args, model, train_dataloader, valid_dataloader, optimizer, valid_terms, device, eval_first=initialize_from_existing_ckpt,train_dataset=train_dataset)


