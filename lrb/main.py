from get_args import get_parser
import torch
import json
from transformers import BertTokenizerFast
from dataset import *
from train import *
import random
import numpy as np
import torch.nn as nn
from model import *
from train_all_dataset import *

parser = get_parser()
args = parser.parse_args()
print(args)

# 设置随机种子
def set_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
set_seed(args.seed)

if args.lrb_num == 2:
    model = LRBModel.from_pretrained(args.model_name, num_labels=2, max_length = args.max_length)
elif args.lrb_num ==1:
    model = LRBOneModel.from_pretrained(args.model_name, num_labels=2, max_length = args.max_length)
model = nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


tk = BertTokenizerFast.from_pretrained(args.model_name)


train_sents, train_labels = read_from_file(args.train_path)
valid_sents, valid_labels = read_from_file(args.valid_path)
train_terms = []
valid_terms = read_terms(args.valid_terms_path)

train_dataset = LRBTrainDataset(train_sents,train_labels,train_terms,tk, args)
valid_dataset = LRBDataset(valid_sents,valid_labels,valid_terms,tk, args)
# from IPython import embed; embed()

train_dataloader = td.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn,pin_memory=True, shuffle=True, num_workers=args.num_workers)
valid_dataloader = td.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=valid_dataset.collate_fn,num_workers=args.num_workers)


from torch.optim import AdamW
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,
                    lr=args.lr,
                    eps=args.eps
                    )

train_model(args, model, train_dataloader, valid_dataloader, optimizer, valid_terms, device,train_dataset = train_dataset)

