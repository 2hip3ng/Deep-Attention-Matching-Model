python train.py \
      --data /data/wangzhipeng05/DAMM_data/data \
      --output /model/wangzhipeng05/DAMM_output/quora \
      --task quora \
      --labels [0,1] \
      --num_hidden_layers 4 \
      --per_gpu_train_batch_size 256 \
      --per_gpu_eval_batch_size 256 \
      --per_gpu_test_batch_size 256 \
      --logging_steps 100 \
      --save_steps 500  \
      --eval_steps 100 \
      --num_train_epochs 80 \
      --learning_rate 0.0002 \
      --max_seq_length_a 25 \
      --max_seq_length_b 25 
# GPU 2
#  
