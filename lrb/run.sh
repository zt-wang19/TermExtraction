
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=6 python3 main.py \
    --ckpt_name ./ckpts/strict/lrb_jian12_strict_{}.pt \
    --lrb_type fewneg \
    --lrb_num 2 \
    --batch_size 128 \
    --seed 49297 \
    --data_split 20 \
    --train_path /home/wangzhitong/data2/all_train_addc_strict_bieos_jian12.txt \
    --valid_path /home/wangzhitong/data2/small_valid_addc_bieos.txt \
    --valid_terms_path /home/wangzhitong/data2/small_valid_terms.txt 