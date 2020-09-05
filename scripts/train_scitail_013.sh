python train.py \
      --data /data/wangzhipeng05/DAMM_data/data \
      --output /model/wangzhipeng05/DAMM_output/scitail \
      --task scitail \
      --labels [0,1] \
      --num_hidden_layers 4 \
      --per_gpu_train_batch_size 256 \
      --per_gpu_eval_batch_size 256 \
      --per_gpu_test_batch_size 256 \
      --logging_steps 30 \
      --save_steps 500 \
      --eval_steps 30 \
      --num_train_epochs 100 \
      --max_seq_length_a 30 \
      --max_seq_length_b 30 \
      --lr_decay_rate 1.0 \
      --learning_rate 0.00015 \
      --use_smooth True
# GPU 2
# 84.4  