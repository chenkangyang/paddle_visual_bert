export DATA_DIR=./X_NLVR/
export LOG_DIR=./logs/nlvr2
unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus "6" --log_dir $LOG_DIR NLVR2/run_nlvr2.py \
    --input_dir $DATA_DIR \
    --output_dir $LOG_DIR \
    --task_name nlvr2 \
    --model_type visualbert \
    --model_name_or_path visualbert-nlvr2 \
    --batch_size 16 \
    --learning_rate 5e-6 \
    --save_steps 200 \
    --num_train_epochs 1