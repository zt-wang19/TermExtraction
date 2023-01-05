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
class Args:
    train_path = '/data2/private/wzt/small_train_addc.jsonl'

args = Args()

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

from testsample import *
testsamp = Sample(sample_probs_funcs[2])

tk = BertTokenizerFast.from_pretrained(args.model_name)
train_data = []
with open(args.train_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f,desc='read train data'):
        train_data.append(json.loads(line))



