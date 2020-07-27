python train.py \
      --data /data/wangzhipeng05/DAMM_data/data \
      --output /model/wangzhipeng05/DAMM_output/snli \
      --task snli \
      --labels [0,1,2] \
      --num_hidden_layers 4 \
      --per_gpu_train_batch_size 256 \
      --per_gpu_eval_batch_size 256 \
      --per_gpu_test_batch_size 256 \
      --logging_steps 100 \
      --save_steps 500 \
      --eval_steps 200 \
      --seed 42 \
      --lr_decay_steps 60000 \
      --warmup_steps 5000 \
      --num_train_epochs 60
# 88.50
