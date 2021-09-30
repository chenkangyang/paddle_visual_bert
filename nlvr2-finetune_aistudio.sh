!python -m paddle.distributed.launch --gpus "0" --log_dir ./logs/nlvr2 NLVR2/run_nlvr2.py \
    --input_dir ./X_NLVR/ \
    --output_dir ./logs/nlvr2 \
    --task_name nlvr2 \
    --model_type visualbert \
    --model_name_or_path checkpoint/paddle_visualbert/visualbert-nlvr2-pre \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --save_steps 5000 \
    --num_train_epochs 10