import random

id2label = ['O','B','I','E','S']
label2id = {label:i for i,label in enumerate(id2label)}

def random_split_train_valid_test(all_sents, all_terms, valid_ratio, test_ratio, seed=42):
    random.seed(seed)
    assert len(all_sents) == len(all_terms)
    idx = [i for i in range(len(all_terms))]
    random.shuffle(idx)
    train_idx = idx[:int(len(all_sents)*(1-valid_ratio-test_ratio))]
    valid_idx = idx[int(len(all_sents)*(1-valid_ratio-test_ratio)):int(len(all_sents)*(1-test_ratio))]
    test_idx = idx[int(len(all_sents)*(1-test_ratio)):]
    train_sents = [list(sent['text']) for i in train_idx  for sent in all_sents[i]]
    train_labels = [sent['bieo_seq'] for i in train_idx  for sent in all_sents[i]]
    valid_sents = [list(sent['text']) for i in valid_idx  for sent in all_sents[i]]
    valid_labels = [sent['bieo_seq'] for i in valid_idx  for sent in all_sents[i]]
    test_sents = [list(sent['text']) for i in test_idx  for sent in all_sents[i]]
    test_labels = [sent['bieo_seq'] for i in test_idx  for sent in all_sents[i]]
    train_terms = list(set([x for i in train_idx for x in all_terms[i]]))
    valid_terms = list(set([x for i in valid_idx for x in all_terms[i]]))
    test_terms = list(set([x for i in test_idx for x in all_terms[i]]))
    return train_sents, train_labels, valid_sents, valid_labels, test_sents, test_labels, train_terms, valid_terms, test_terms

