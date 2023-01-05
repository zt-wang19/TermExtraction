
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=7 python3 main.py \
    --ckpt_name ./ckpts/strict/ner_strict_{}.pt \
    --batch_size 128 \
    --seed 42 \
    --data_split 20 \
    --train_path /home/wangzhitong/data2/all_train_addc_strict_bieos.txt \
    --valid_path /home/wangzhitong/data2/small_valid_addc_bieos.txt \
    --valid_terms_path /home/wangzhitong/data2/small_valid_terms.txt 