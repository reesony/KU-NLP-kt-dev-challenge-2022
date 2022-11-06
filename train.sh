CUDA_VISIBLE_DEVICES=0 python3 train.py \
--model_name_or_path KETI-AIR/ke-t5-base-ko \
--output_dir ./models/base_model \
--train_file train.csv \
--validation_file validation.csv \
--overwrite_output_dir \
--preprocessing_num_workers 8 \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 4 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--warmup_ratio 0.05 \
--weight_decay 5e-6 \
--max_length 256 \
--logging_dir ./logs \
--logging_steps 100 \
--save_strategy epoch \
--evaluation_strategy epoch \
--project_name kt-dev-challenge \
--load_best_model_at_end \
--metric_for_best_model f1