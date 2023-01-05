from get_args import get_parser
import torch
from transformers import BertTokenizerFast
from dataset import *
from train import *
from model import *
import random
import numpy as np
import torch.nn as nn
from train_all_dataset import NERTrainDataset

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

if args.use3:
    model = NERThreeModel.from_pretrained(args.model_name, num_labels=len(id2label),max_length=args.max_length)    
else:
    model = NERModel.from_pretrained(args.model_name, num_labels=len(id2label))
model = nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


tk = BertTokenizerFast.from_pretrained(args.model_name)


train_sents, train_labels = read_from_file(args.train_path)
valid_sents, valid_labels = read_from_file(args.valid_path)
train_terms = []
valid_terms = read_terms(args.valid_terms_path)

train_dataset = NERTrainDataset(train_sents,train_labels,train_terms,tk,args)
valid_dataset = NERDataset(valid_sents,valid_labels,valid_terms,tk, args)
# test_dataset = NERDataset(test_sents,test_labels,test_terms,tk, args)


train_dataloader = td.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.ner_collate_fn,pin_memory=True, shuffle=True, num_workers=args.num_workers)
valid_dataloader = td.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=train_dataset.ner_collate_fn,num_workers=args.num_workers)
# test_dataloader = td.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=ner_collate_fn,num_workers=args.num_workers,timeout=10)


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

# evaluate(model, valid_dataloader,valid_terms, device)
train_model(args, model, train_dataloader, valid_dataloader, optimizer, valid_terms, device,train_dataset = train_dataset)




