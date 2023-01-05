import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps')
    parser.add_argument('--max_length', type=int, default=128, help='max length')
    # model_name
    parser.add_argument('--model_name', type=str, default='hfl/chinese-roberta-wwm-ext', help='model name')
    # dataset_size
    parser.add_argument('--dataset_size', type=str, default='10000',choices = ['10000','full'], help='dataset size')
    # valid_ratio
    parser.add_argument('--valid_ratio', type=float, default=0.05, help='valid ratio')
    # test_ratio
    parser.add_argument('--test_ratio', type=float, default=0.05, help='test ratio')
    # seed
    parser.add_argument('--seed', type=int, default=42, help='seed')
    # chunk_size
    parser.add_argument('--chunk_size', type=int, default=1000, help='chunk size')
    # ckpt_name
    parser.add_argument('--ckpt_name', type=str, default='', help='ckpt name')
    # num_worker
    parser.add_argument('--num_workers', type=int, default=32, help='num worker')
    # N
    parser.add_argument('--N', type=int, default=15, help='N')
    # train_path
    parser.add_argument('--train_path', type=str, default='', help='train path')
    # valid_path
    parser.add_argument('--valid_path', type=str, default='', help='valid path')
    parser.add_argument('--finetuned_ckpt', type=str,
    default="")
    parser.add_argument('--sample_probs_choice', type=int, default=1)
    parser.add_argument('--neg_ratio_choice', type=int, default=1 )
    parser.add_argument('--context_size', type=int,default=3)
    parser.add_argument('--log_dir', type=str, default="../logs" )
    parser.add_argument('--uid', type=str, default="")
    parser.add_argument('--early_stop_epoch', type=int, default = 10000)
    parser.add_argument('--use_two_classifier', type=int, default=0)
    parser.add_argument('--data_split', type=int, default=1)
    parser.add_argument('--valid_terms_path', type=str, default='', help='valid term path')
    parser.add_argument('--lrb_type', type=str, default='fewneg',choices = ['fewneg','allneg'], help='lrb type')
    parser.add_argument('--lrb_num', type=int, default=2, help='lrb num')
    # jian12
    parser.add_argument('--jian12', type=int, default=0, help='jian12')
    return parser