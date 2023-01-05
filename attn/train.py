import torch
from tqdm import tqdm
import datetime
import time
import numpy as np
from utils import *
import os
import torch.utils.data as td
from dataset import NegDataset


def compute_score(extracted_terms, gold_terms):
    epsilon = 1e-8
    extracted_terms = set(extracted_terms)
    gold_terms = set(gold_terms)
    exlen = len(extracted_terms)
    goldlen = len(gold_terms)
    intersectlen = len(extracted_terms.intersection(gold_terms))
    precision = intersectlen/(exlen+epsilon)
    recall = intersectlen/(goldlen+epsilon)
    f1 = 2*precision*recall/(precision+recall+epsilon)
    return precision, recall, f1

# def get_ngram_idxes(n,max_len):
#     res = []
#     for i in range(max_len):
#         for j in range(n):
#             if i+j < max_len:
#                 res.append((i,i+j))
#     return res


inspect_sents = [
    "机器学习的基准在不断进化。",
    "机器学习基准在不断进化。",
    "另取醋酸可的松对照品适量。",
    "它使⽤如下参数：返回新的套接字描述符，出错返回-1。",
    "如果路由器的工作仅仅是在子网与子网间、⽹络与⽹络间交换数据包的话，我们可能会买到⽐今天便宜得多的路由器。",
    "尼俄伯号小巡洋舰是德意志帝国于1890年代末至1900年代初所建造的十艘瞪羚级小巡洋舰的二号舰，以希腊神话人物尼俄伯命名。",
    "美国电影艺术与科学学院向曾代表马龙·白兰度领取奥斯卡最佳男主角奖而遭攻击的原住民运动者萨钦·小羽毛（图）道歉",
    "系统架构的关键因素是数据结构而非算法的见解，导致了多种形式化的设计方法与编程语言的出现。绝大多数的语言都带有某种程度上的模块化思想，透过将数据结构的具体实现封装隐藏于用户界面之后的方法，来让不同的应用程序能够安全地重用这些数据结构。C++、Java、Python等面向对象的编程语言可使用类 (计算机科学)来达到这个目的。",
    "在计算机科学中，链表（Linked list）是一种常见的基础数据结构，是一种线性表，但是并不会按线性的顺序存储数据，",
    "散列表（Hash table，也叫哈希表），是根据键（Key）而直接访问在内存储存位置的数据结构。也就是说，它通过计算出一个键值的函数，将所需查询的数据映射到表中一个位置来让人访问，这加快了查找速度。这个映射函数称做散列函数，存放记录的数组称做散列表。",
    "大多数数据结构都由数列、记录、可辨识联合、引用等基本类型构成。举例而言，可为空的引用（nullable reference）是引用与可辨识联合的结合体，而最简单的链式结构链表则是由记录与可空引用构成。",
    "陛下之UC-23号艇（德语：SM UC 23[注 1]）是德意志帝国海军于第一次世界大战期间建造的一艘UC-II型近岸布雷潜艇或称U艇。",
    "1915年秋天，由于中立国美国的干预，U艇战几乎陷入停顿，导致德国广泛开展《国际法》所允许的水雷战，从而使布雷潜艇的需求量相应增加。",
    "但世卫干事长谭德塞对这份备受关注的报告公开表达了担忧",
    "1月1日，在Omicron变异株激增的情况下，欧洲的病例数超过了1亿例",
    "严重疾病的最强风险因素是肥胖、糖尿病并发症、焦虑症和疾病总数",
    "家具是由材料、结构、外观形式和功能四种因素组成，其中功能是先导，是推动家具发展的动力；结构是主干，是实现功能的基础。",
    "家具是指人类维持正常生活、从事生产实践和开展社会活动必不可少的器具设施大类。家具也跟随时代的脚步不断发展创新，到如今门类繁多，用料各异，品种齐全，用途不一。是建立工作生活空间的重要基础。",
    "处暑，即为“出暑”，是炎热离开的意思。“三伏天”涉及小暑、大暑、立秋、处暑四个节气，这时三伏已过或近尾声，初秋炎热将结束。处暑的到来同时也意味着进入干支历申月的下半月。",
    "处暑，北斗七星的斗柄是指向西南方向（戊位），太阳到达黄经150°时，交节时间点在公历8月23日前后"
]


def evaluate(model, dataloader, gold_terms, device, args, run_false_vis=False, uid="", log_dir=""):
    model.eval()

    with torch.no_grad():
        for s in inspect_sents[:4]:
            model.inference_one_sent(s)

    total_loss = 0
    extracted_terms = []
    # ngram_idxes = get_ngram_idxes(args.N,args.max_length-2)

    all_final_logits = []
    all_loss_labels = []

    # for visualizing the error cases
    all_batch_input_ids = []
    all_in_batch_ids = []
    all_term_labels = []
    all_batch_offset = 0

    with torch.no_grad():
        iterator = tqdm(enumerate(dataloader),
                        total=len(dataloader), ascii=True)
        for step, batch in iterator:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2]
            b_real_lens = batch[3]

            # b_idxes = batch[3]
            if len(b_labels) == 0:
                continue

            final_logits, loss_labels, in_batch_ids, term_labels = model.inference(input_ids=b_input_ids,
                                                                                   token_type_ids=None,
                                                                                   attention_mask=b_input_mask,
                                                                                   labels=b_labels,
                                                                                   real_lens=b_real_lens,
                                                                                   debug=True,
                                                                                   )

            # for visualizing the error cases
            in_batch_ids += all_batch_offset
            all_batch_offset += b_input_ids.size(0)
            all_in_batch_ids.append(in_batch_ids)
            all_term_labels.append(term_labels)
            all_batch_input_ids.append(b_input_ids)

            all_final_logits.append(final_logits)
            all_loss_labels.append(loss_labels)

    all_batch_input_ids = torch.cat(all_batch_input_ids, dim=0).cpu().numpy()
    all_in_batch_ids = torch.cat(all_in_batch_ids, dim=0).cpu().numpy()
    all_term_labels = torch.cat(all_term_labels, dim=0).cpu().numpy()

    all_final_logits = np.concatenate(all_final_logits, axis=0)
    all_loss_labels = np.concatenate(all_loss_labels, axis=0)

    f1, precision, recall, false_negative, false_positive, ave_prec = model.prediction(
        all_final_logits, all_loss_labels)

    if run_false_vis:
        # from IPython import embed; embed()
        get_false_neg_term(false_negative=false_negative, all_batch_input_ids=all_batch_input_ids,
                           all_term_labels=all_term_labels, all_in_batch_ids=all_in_batch_ids, tokenizer=model.tk, file=f"{log_dir}/false_neg_analysis_{uid}.txt"
                           )
        # from IPython import embed; embed()
        get_false_neg_term(false_negative=false_positive, all_batch_input_ids=all_batch_input_ids,
                           all_term_labels=all_term_labels, all_in_batch_ids=all_in_batch_ids, tokenizer=model.tk, file=f"{log_dir}/false_pos_analysis_{uid}.txt"
                           )

    print("precision: {0:.3f}".format(precision*100))
    print("recall: {0:.3f}".format(recall*100))
    print("f1: {0:.3f}".format(f1*100))
    print("ave_prec: {0:.3f}".format(ave_prec*100))

    # print("f1 {}, precision {}, recall {}".format(f1, precision, recall))
    return precision, recall, f1, ave_prec


def get_false_neg_term(false_negative, all_batch_input_ids, all_term_labels, all_in_batch_ids, tokenizer, file):
    fout = open(file, 'w')
    false_negative = false_negative[0]
    last_in_batch_id = -1
    for i in false_negative:
        in_batch_id = all_in_batch_ids[i]
        if in_batch_id != last_in_batch_id:
            print("\n\n", file=fout)
            sentence = tokenizer.decode(
                all_batch_input_ids[in_batch_id], skip_special_tokens=True)
            print(sentence, file=fout)
            last_in_batch_id = in_batch_id

        term_labels = all_term_labels[i]
        tail = term_labels[np.where(term_labels != -1)[0].max()]
        head = term_labels[0]
        term_id = all_batch_input_ids[in_batch_id, head:tail+1]
        term = tokenizer.convert_ids_to_tokens(term_id)
        print(" ".join(term), end=";", file=fout)
    fout.close()
    print("analysis false done!")


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_model(args, model, train_dataloader, valid_dataloader, optimizer, valid_terms, device, eval_first,train_dataset=None):

    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # mostly contains scores about how the training went for each epoch
    training_stats = []

    # total training time
    total_t0 = time.time()

    if eval_first:
        print("eval first")
        evaluate(model, valid_dataloader, valid_terms, device, args)
    else:
        print("not eval first")

    print('\033[1m'+"================ Model Training ================"+'\033[0m')
    best_metric = -1
    metric_name = 'accuracy'
    for epoch_i in range(0, args.epochs):

        print('ckpt_name:', args.ckpt_name.format(epoch_i+1))
        print(
            '\033[1m'+'======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs)+'\033[0m')

        t0 = time.time()

        # summed training loss of the epoch
        total_train_loss = 0

        # model is being put into training mode as mechanisms like dropout work differently during train and test time
        model.train()

        for split_i in range(args.data_split):
            # iterrate over batches
            # if train_dataset:
            if split_i > 0:
                train_dataset.update_split(split_i)
            train_dataloader = td.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,collate_fn=NegDataset.collate_fn)
            iterator = tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader), ascii=True,desc = f'training data split {split_i}')
            for step, batch in iterator:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2]  # .to(device)
                b_real_lens = batch[3]
                b_input_sents = batch[4]
                if len(batch) > 5:
                    b_cnegs = batch[5]
                if len(b_labels) == 0:
                    continue
                output = model(input_ids=b_input_ids,
                               input_sents=b_input_sents,
                               token_type_ids=None,
                               attention_mask=b_input_mask,
                               labels=b_labels,
                               real_lens=b_real_lens,
                               cnegs=b_cnegs
                               )

                loss = output[1]
                # iterator.set_description("loss: {:.5f}".format(loss.item()))

                loss.backward()
                total_train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                for s in inspect_sents[:4]:
                    model.inference_one_sent(s)
            ckpt_name = args.ckpt_name.format(str(epoch_i+1)+'_split_'+str(split_i))
            print('saving ckpt to', ckpt_name)
            if not os.path.exists(f"ckpts/{args.uid}/"):
                os.mkdir(f"ckpts/{args.uid}/")

            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(),
                           f"ckpts/{args.uid}/{ckpt_name}")
            else:
                torch.save(model.state_dict(),
                           f"ckpts/{args.uid}/{ckpt_name}")

        # avg loss over all batches
        avg_train_loss = total_train_loss / len(train_dataloader)

        # training time of this epoch
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # VALIDATION
        print("evaluate")
        precision, recall, f1, ave_prec = evaluate(
            model, valid_dataloader, valid_terms, device, args)
        # precision,recall,f1 = 0,0,0

        cur_stats = {
            'neg_ratio': args.neg_ratio_choice,
            'sample_probs': args.sample_probs_choice,
            'learning_rate': args.lr,
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Training Time': training_time,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ave_prec': ave_prec,
            'uid': args.uid,
        }

        print(cur_stats)
        with open(args.log_dir + "/all_results.txt", 'a') as fout:
            print(cur_stats, file=fout)

        if ave_prec > best_metric:
            best_metric = ave_prec
            es = 0
            best_ckpt_name = args.ckpt_name.format(epoch_i+1)
            if not os.path.exists(f"ckpts/{args.uid}/"):
                os.mkdir(f"ckpts/{args.uid}/")

            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(),
                           f"ckpts/{args.uid}/{best_ckpt_name}")
            else:
                torch.save(model.state_dict(),
                           f"ckpts/{args.uid}/{best_ckpt_name}")
        else:
            es += 1
            print("Counter {} of 3".format(es))

            if es > args.early_stop_epoch:
                print("Early stopping with best_metric: ", best_metric,
                      "and val_metric for this epoch: ", f1, "...")
                break

    print("\n\nTraining complete!")
    print("Total training took {:} (h:mm:ss)".format(
        format_time(time.time()-total_t0)))

    model.load_state_dict(torch.load(f"ckpts/{args.uid}/{best_ckpt_name}"))
    evaluate(model, valid_dataloader, valid_terms, device, args,
             run_false_vis=False, uid=args.uid, log_dir=args.log_dir)

    return training_stats
