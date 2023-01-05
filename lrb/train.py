import torch
from tqdm import tqdm
import datetime
import time
import numpy as np
from utils import *
import torch.utils.data as td

def compute_score(extracted_terms,gold_terms):
    epsilon = 1e-8
    extracted_terms = set(extracted_terms)
    gold_terms = set(gold_terms)
    exlen = len(extracted_terms)
    goldlen = len(gold_terms)
    intersectlen = len(extracted_terms.intersection(gold_terms))
    precision = intersectlen/(exlen+epsilon)
    recall = intersectlen/(goldlen+epsilon)
    f1 = 2*precision*recall/(precision+recall+epsilon)
    return precision,recall,f1

def get_ngram_idxes(n,max_len):
    res = []
    for i in range(max_len):
        for j in range(n):
            if i+j < max_len:
                res.append((i,i+j))
    return res
def evaluate(model, dataloader, gold_terms,device,args):
    model.eval()
    total_loss = 0
    extracted_terms = []
    # ngram_idxes = get_ngram_idxes(args.N,args.max_length-2)
    with torch.no_grad():
        for idx,batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            input_ids, attention_mask, labels, sents = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids = input_ids, 
                attention_mask = attention_mask)
            bs,seqlen,hs = outputs[0].shape
            logits = outputs[0].detach().cpu().numpy().argmax(axis=-1).reshape(bs,2,seqlen//2)
            for logit,sent in zip(logits,sents):
                sent_len = min(len(sent),args.max_length-2)
                sent_ngram_idxes = get_ngram_idxes(args.N,sent_len)
                for i,j in sent_ngram_idxes:
                    if logit[0,i]==1 and logit[1,j+1]==1:
                        extracted_terms.append(''.join(sent[i:j+1]))
    precision,recall,f1 = compute_score(extracted_terms,gold_terms)
    return precision,recall,f1,extracted_terms

def evaluate1(model, dataloader, gold_terms,device,args):
    model.eval()
    extracted_terms = []
    with torch.no_grad():
        for idx,batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            input_ids, attention_mask, labels, sents = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids = input_ids, 
                attention_mask = attention_mask)
            bs,seqlen,hs = outputs[0].shape
            logits = outputs[0].detach().cpu().numpy().argmax(axis=-1).reshape(bs,2,seqlen//2)
            for logit,sent in zip(logits,sents):
                logit = logit[:,1:-1]
                sent_len = min(len(sent),args.max_length-2)
                sent_ngram_idxes = get_ngram_idxes(args.N,sent_len)
                for i,j in sent_ngram_idxes:
                    if j+1< args.max_length-2:
                        if logit[0,i]==1 and logit[1,j+1]==1:
                            extracted_terms.append(''.join(sent[i:j+1]))
    precision,recall,f1 = compute_score(extracted_terms,gold_terms)
    return precision,recall,f1,extracted_terms

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_model(args, model, train_dataloader, valid_dataloader, optimizer, valid_terms, device,train_dataset = None):

    # mostly contains scores about how the training went for each epoch
    training_stats = []

    # total training time
    total_t0 = time.time()

    print('\033[1m'+"================ Model Training ================"+'\033[0m')
    best_metric = -1
    metric_name = 'accuracy'
    for epoch_i in range(0, args.epochs):
        epochstr = f'epoch_{epoch_i+1}'
        ckpt_name = args.ckpt_name.format(epochstr)
        print('ckpt_name:',ckpt_name)
        print(
            '\033[1m'+'======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs)+'\033[0m')

        t0 = time.time()

        # summed training loss of the epoch
        total_train_loss = 0

        # model is being put into training mode as mechanisms like dropout work differently during train and test time
        model.train()

        # iterrate over batches
        for split_i in range(args.data_split):
            if split_i > 0:
                train_dataset.update_split(split_i)
                train_dataloader = td.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,collate_fn=train_dataset.collate_fn)
            iterator = tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader), ascii=True,desc = f'training data split {split_i}')
            for step, batch in iterator:
                b_input_ids = batch[0].to(device)   
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                b_sents = batch[3]

                output = model(input_ids=b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                                    
                loss = output[1].mean()
                
                loss.backward()
                # print(loss.item())
                total_train_loss += loss.item()
                # if step % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            ckpt_name = args.ckpt_name.format(epochstr)[:-3] + f'_split_{split_i}.pt'
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(),ckpt_name)
            else:
                torch.save(model.state_dict(), ckpt_name)

        # avg loss over all batches
        avg_train_loss = total_train_loss / len(train_dataloader)

        # training time of this epoch
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # VALIDATION
        print("evaluate")
        if args.lrb_num==2:
            precision,recall,f1,_ = evaluate(model, valid_dataloader, valid_terms, device,args)
        elif args.lrb_num == 1:
            precision,recall,f1,_ = evaluate1(model, valid_dataloader, valid_terms, device,args)
        print("precision: {0:.2f}".format(precision*100))
        print("recall: {0:.2f}".format(recall*100))
        print("f1: {0:.2f}".format(f1*100))

        cur_stats = {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Training Time': training_time,
            }

        print(cur_stats)

        # if isinstance(model, torch.nn.DataParallel):
        #     torch.save(model.module.state_dict(),ckpt_name)
        # else:
        #     torch.save(model.state_dict(), ckpt_name)

        if f1 > best_metric:
            best_metric = f1
            es = 0
        else:
            es += 1
            print("Counter {} of 3".format(es))

            if es > 2:
                print("Early stopping with best_metric: ", best_metric,
                      "and val_metric for this epoch: ", f1 , "...")

    print("\n\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(
        format_time(time.time()-total_t0)))

    return training_stats