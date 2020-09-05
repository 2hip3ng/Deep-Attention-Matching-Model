python train.py \
      --data /data/wangzhipeng05/DAMM_data/data \
      --output /model/wangzhipeng05/DAMM_output/snli \
      --task snli \
      --labels [0,1,2] \
      --num_hidden_layers 5 \
      --per_gpu_train_batch_size 200 \
      --per_gpu_eval_batch_size 200 \
      --per_gpu_test_batch_size 200 \
      --logging_steps 100 \
      --save_steps 500 \
      --eval_steps 200 \
      --learning_rate 0.00015
# GPU 2

