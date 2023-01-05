NEGRATIO=5
SAMPLEPROBS=2
GPUID=5

neg=$NEGRATIO
sam=$SAMPLEPROBS
uniq="strict_neg_"${neg}"sam"${sam}
echo $uniq
CUDA_VISIBLE_DEVICES=$GPUID python main.py --batch_size 128 --train_path /data2/private/wzt/all_train_addc_strict.jsonl  --valid_path /data2/private/wzt/small_dev_addc_no23common.jsonl  --ckpt_name ${uniq}_epoch{}.pt --sample_probs_choice $sam --neg_ratio_choice $neg --uid $uniq --early_stop_epoch 10000 --data_split 20 --valid_terms_path /home/wangzhitong/data2/small_dev_addc_no23common_terms.txt 
# >> "../logs/${uniq}.log" 2>&1;


