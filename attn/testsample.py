import random
class Sample:
    def __init__(self,probs) -> None:
        self.probs = probs
        self.max_term_size = 128
        pass
    
    def get_negative(self, sent, label, sent_len,cneg, neg_ratio=2, least_neg_num=2, max_term_size=15, context_size=3, resample=False,cnegs= None):
        negs = []
        for neg_id in range(max(neg_ratio*len(label), least_neg_num)):
            sent_len = min(self.max_term_size, sent_len+1)
            neg = self.sample(sent, label, sent_len, probs=self.probs,cneg = cneg,max_term_size=max_term_size)
            negs.append(neg)
        return negs

    def sample(self, common_word, label, sent_len, probs={"random":0.5,"overlap":0.3,"concate":0.2},cneg=None,max_term_size = 10):
        s = random.uniform(0, 1)
        # from IPython import embed; embed(header="in sample")
        for key, p in probs.items():
            s -= p
            if s <= 0:
                if key == "random":
                    neg = self.random_sample(label, sent_len)
                elif key == "overlap":
                    neg = self.overlap_sample(label, sent_len)
                elif key == "concate":
                    neg = self.concate_sample(label, sent_len)
                elif key == "common":
                    neg = self.common_sample(common_word, label, sent_len,cneg,max_term_size)
                return neg

    def common_sample(self, common_word, label, sent_len,neg,max_term_size):

        total = len(neg)
        if total == 0:
            return self.random_sample(label, sent_len)

        ok = False
        count = 0
        res=None
        while not ok:
            count += 1
            if count > 3:
                return self.random_sample(label, sent_len)
            s = random.randint(0, total-1)
            # neg = common_word[s]
            res = neg[s]
            if neg not in label and res[1]-res[0]<max_term_size:
                # print('res:',res)
                ok = True
            # else:
                # print('else:',res)
        return 'c',res

        

    def random_sample(self, label, sent_len, window_max=10):
        window_max = min(window_max, self.max_term_size -1)
        ok = False
        cnt = 0
        while not ok:
            
            s = random.randint(1, sent_len-1)
            window = random.randint(1, window_max+1)
            t = s
            t = random.randint(max(1, s-window), min(sent_len-1, s+window+1))
            neg = (s, t) if s <= t else (t, s)
            if neg not in label or cnt > 10:
                ok = True
            cnt+=1
        return 'r',neg

    def overlap_sample(self, label, sent_len):
        label_num = len(label)
        if label_num == 0:
            return self.random_sample(label, sent_len) # no label, sample randomly
        
        labelid = random.randint(0, label_num)
        neg = label[labelid]
        count = 0 
        while neg in label:
            count += 1
            if count > 2:
                return self.random_sample(label, sent_len)
            labelid = random.randint(0, label_num)
            neg = label[labelid]
            random_float = random.uniform(0,1)
            if random_float<0.3: # add to both side
                left_move = random.randint(1, 2)
                right_move = random.randint(1, 2)
                s = max(1, neg[0] - left_move) # 1: not include the first one [CLS]
                t = min(sent_len - 2, neg[1] + right_move) # -2: not include the last one [SEP]
                neg = (s, t)
            elif random_float<0.65: # add to left
                left_move = random.randint(1, 2)
                s = max(1, neg[0] - left_move)
                t = neg[1]
                neg = (s, t)
            else: # add to right
                right_move = random.randint(1, 2)
                s = neg[0]
                t = min(sent_len - 2, neg[1] + right_move) # -2: not include the last one [SEP]
                neg = (s, t)
        if t - s + 1 > self.max_term_size:
            return self.random_sample(label, sent_len) # too long, sample randomly
        return neg

    def concate_sample(self, label, sent_len):
        label_num = len(label)
        if label_num <= 1:
            return self.random_sample(label, sent_len) # only one/none label, use random sample
        count = 0
        while True:
            count += 1
            if count > 2: # reduce to random sample
                return self.random_sample(label, sent_len)
            indices = random.choice(label_num, 2, replace=False)
            index_left, index_right= (indices[0], indices[1]) if indices[0]<indices[1] else (indices[1], indices[0])
            pos_left, pos_right = label[index_left], label[index_right]
            if pos_left[1] + 5 > pos_right[0] and (pos_left[0], pos_right[1]) not in label:
                break
        neg = (pos_left[0], pos_right[1])
        if neg[1] - neg[0] + 1 > self.max_term_size:
            return self.random_sample(label, sent_len) # too long, sample randomly
        return 'co', neg

        