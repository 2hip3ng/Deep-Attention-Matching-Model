python ../../train.py \
      --data ./../../data \
      --output /model/wangzhipeng05/DAMM_output/snli \
      --task snli \
      --output /data/snli \
      --per_gpu_train_batch_size 1 \
      --per_gpu_eval_batch_size 1 \
      --per_gpu_test_batch_size 1 \
      --logging_steps 1 \
      --save_steps 1  

