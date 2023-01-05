import torch.utils.data as td
import json
from tqdm import tqdm
import random
import torch


def get_sents_labels(path, max_len):
    max_len = max_len-2
    data = []
    with open(path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            if isinstance(tmp, str):
                tmp = eval(tmp)
            data.append(tmp)
    sents = [list(sent['text'])[:max_len] for text in data for sent in text]
    labels = [list(sent['bieo_seq'])[:max_len]
              for text in data for sent in text]
    return sents, labels


def get_ngrams(sents, labels, max_len):
    print('Getting ngrams')
    pos_ngrams = []
    for sent, label in zip(sents, labels):

        pos_idxes = []
        beg = -1
        for i, l in enumerate(label):
            if l == 'B':
                beg = i
            elif l == 'I':
                pass
            elif l == 'E':
                if beg != -1:
                    pos_idxes.append((beg, i))
                beg = -1
            elif l == 'S':
                pos_idxes.append((i, i))
            elif l == 'O':
                # pass
                beg = -1
        pos_ngrams.append([sent, pos_idxes])
    return pos_ngrams


def get_terms(pos_ngrams):
    print('Getting terms')
    terms = []
    for sent, pos_idxes in tqdm(pos_ngrams):
        for beg, end in pos_idxes:
            terms.append(''.join(sent[beg:end+1]))
    return set(terms)

def open_train_ngrams(ngrams):
    print('Opening train ngrams')
    sents = []
    idxes = []
    labels = []
    for sent, pos_idxes, neg_idxes in ngrams:
        sents.append(sent)

        sent_idxes = pos_idxes+neg_idxes
        sent_idxes = [(i+1, j+1) for i, j in sent_idxes]
        idxes.append(sent_idxes)

        labels.append([1]*len(pos_idxes)+[0]*len(neg_idxes))
    return sents, idxes, labels


def get_ngram_idx(length, N):
    idxes = []
    for i in range(length):
        for j in range(i, length):
            if j-i < N:
                idxes.append((i+1, j+1))
    return idxes


def get_valid_ngrams(sents):
    print('Getting valid ngrams')
    N = 8
    idxes = []
    labels = []
    for sent in sents:
        idxes.append(get_ngram_idx(len(sent), N))
        labels.append([1]*len(idxes[-1]))
    return sents, idxes, labels

class NegDataset(td.Dataset):
    def __init__(self, data, tk, args,init0 = True):
        self.all_data = data
        self.tk = tk
        self.max_length = args.max_length
        self.data_split = args.data_split
        self.data_len = len(data)

        x = (len(data)//self.data_split)+1
        self.split_idx = [(x*i,x*(i+1))  for i in range(self.data_split)]
        if init0:
            self.update_split(0)

    def update_split(self, split):
        assert split>=0 and split<self.data_split
        split_data = self.all_data[self.split_idx[split][0]:self.split_idx[split][1]]
        sents = [list(data[0]) for data in split_data]
        chunk_size = 16
        # chunk sents into groups of 16
        chunked_sents = [sents[i:i+chunk_size] for i in range(0, len(sents), chunk_size)]
        tmp = []
        for chunk in tqdm(chunked_sents, desc=f'tokenizing split{split}'):
            tmp.append(self.tk(chunk,
                              padding='max_length',
                              truncation=True,
                              max_length=self.max_length,
                              is_split_into_words=True,
                              return_tensors='pt'))
        ed = {'input_ids': torch.cat([item['input_ids'] for item in tmp], dim=0),
              'attention_mask': torch.cat([item['attention_mask'] for item in tmp], dim=0),
              'token_type_ids': torch.cat([item['token_type_ids'] for item in tmp], dim=0)}

        ed['label'] = []
        ed['neg'] = []
        for data in split_data:
            label = data[1]
            label = [(pair[0]+1, pair[1]+1) for pair in label if pair[1] < self.max_length-1]
            neg = [(pair[1]+1, pair[2]+1) for pair in data[2] if pair[2] < self.max_length-1]
            ed['label'].append(label)
            ed['neg'].append(neg)

        real_len = torch.sum(ed['attention_mask'], dim=1).cpu().tolist()
        ed['real_len'] = real_len

        ret = {"input_ids":[], "attention_mask":[], "label":[],  "real_len":[], 'input_sents':[],'neg':[]}

        for s, i, r, l, a, n in tqdm(zip(sents, ed['input_ids'], ed['real_len'], ed['label'], ed['attention_mask'], ed['neg']), desc='removing not aligned'):
            if r == len(s) + 2:
                ret['input_ids'].append(i)
                ret['attention_mask'].append(a)
                ret['label'].append(l)
                ret['real_len'].append(r)
                ret['neg'].append(n)
                sent = self.tk.convert_ids_to_tokens(i)
                common_ids = []
                ret['input_sents'].append(common_ids)
        self.processed_data = ret

    def __getitem__(self, index):
        return self.processed_data['input_ids'][index], self.processed_data['attention_mask'][index], self.processed_data['label'][index], self.processed_data['real_len'][index], self.processed_data['input_sents'][index],self.processed_data['neg'][index]

    def __len__(self):
        return len(self.processed_data['input_ids'])
    

    @staticmethod
    def collate_fn(batch):
        input_ids = [item[0] for item in batch]
        attention_mask = [item[1] for item in batch]
        # labels = [x for item in batch for x in item[2]]
        label = [item[2] for item in batch]
        real_len = [item[3] for item in batch]
        input_sents = [item[4] for item in batch]
        negs = [item[5] for item in batch]
        return torch.vstack(input_ids), torch.vstack(attention_mask), label, real_len, input_sents,negs


class SophDatasetNegSampling(td.Dataset):
    def __init__(self, data, tk, args):
        # from IPython import embed; embed(header = "sophneg_dataset")
        self.all_data = data
        self.tk = tk
        self.max_length = args.max_length
        self.chunk_size = 8
        # self.neg = [item[2] for item in data]


        # self.processed_data = []
        # for data in self.all_data:
        sents = [data[0] for data in self.all_data]
        
        ed = self.tk(sents,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    is_split_into_words=True,
                    return_tensors='pt')
        
        # from IPython import embed; embed(header = "sophneg_dataset")

        ed['label'] = []
        for data in self.all_data:
            label = data[1]
            label = [(pair[0]+1, pair[1]+1) for pair in label if pair[1] < self.max_length-1]
            # neg = 
            ed['label'].append(label)

        real_len = torch.sum(ed['attention_mask'], dim=1).cpu().tolist()
        ed['real_len'] = real_len

        ret = {"input_ids":[], "attention_mask":[], "label":[],  "real_len":[], 'input_sents':[]}

        # self.read_common_word()

        for s, i, r, l, a in tqdm(zip(sents, ed['input_ids'], ed['real_len'], ed['label'], ed['attention_mask']), desc='removing not aligned'):
            if r == len(s) + 2:
                ret['input_ids'].append(i)
                ret['attention_mask'].append(a)
                ret['label'].append(l)
                ret['real_len'].append(r)
                sent = self.tk.convert_ids_to_tokens(i)
                common_ids = []
                # if args.use_common_word
                # for window in range(2,6):
                #     for start in range(1, len(sent)):
                #         tmpngram = ''.join(sent[start:start+window])
                #         if '[PAD]' in tmpngram:
                #             break
                #         if tmpngram in self.common_ngram_set:
                #             common_ids.append((start, start+window-1))
                
                            
                # from IPython import embed; embed(header='True')
                ret['input_sents'].append(common_ids)
            else:
                # print(r, len(s), s, self.tk.decode(i))
                pass
        
        self.processed_data = ret
        # from IPython import embed; embed(header = "sophneg_dataset")


    def read_common_word(self, path=None):
        print("read common words")
        if path is None:
            path = "../common_ngram.xlsx"
        common_ngram = pd.read_excel(path)
        self.common_ngram_set = set(common_ngram['word'].tolist()[:40351])






    def __getitem__(self, index):
        return self.processed_data['input_ids'][index], self.processed_data['attention_mask'][index], self.processed_data['label'][index], self.processed_data['real_len'][index], self.processed_data['input_sents'][index]

    def __len__(self):
        return len(self.processed_data['input_ids'])
    

    @staticmethod
    def collate_fn(batch):
        input_ids = [item[0] for item in batch]
        attention_mask = [item[1] for item in batch]
        # labels = [x for item in batch for x in item[2]]
        label = [item[2] for item in batch]
        real_len = [item[3] for item in batch]
        input_sents = [item[4] for item in batch]
        # sents = [item[4] for item in batch]
        return torch.vstack(input_ids), torch.vstack(attention_mask), label, real_len, input_sents
