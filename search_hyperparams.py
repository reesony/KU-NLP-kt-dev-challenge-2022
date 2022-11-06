import os
from itertools import product

lr_list = [1e-04, 3e-05, 5e-05]
batch_list = [16, 32]
epoch_list = [4, 6]
wd_list = [0.1, 5e-06]

SH_LIST_DIR = os.path.abspath("sh_list")
os.makedirs(SH_LIST_DIR, exist_ok=True)

for index, (lr, b, e, wd) in enumerate(list(product(lr_list, batch_list, epoch_list, wd_list))):
    path = os.path.join(SH_LIST_DIR, f"train_{index + 1}.sh")
    with open(path, "w") as f:
        sh_template = f"""CUDA_VISIBLE_DEVICES=0 python3 train.py \\
    --model_name_or_path /home/work/team02/model/kt-ulm-base \\
    --output_dir ./models/base-lr_{lr}-batch_{b}-epoch_{e}-wd_{wd} \\
    --train_file train_and_valid.csv \\
    --validation_file test.csv \\
    --overwrite_output_dir \\
    --preprocessing_num_workers 8 \\
    --do_train \\
    --do_eval \\
    --learning_rate {lr} \\
    --num_train_epochs {e} \\
    --per_device_train_batch_size {b} \\
    --per_device_eval_batch_size 64 \\
    --gradient_accumulation_steps 1 \\
    --warmup_ratio 0.05 \\
    --weight_decay {wd} \\
    --max_length 256 \\
    --logging_dir ./logs \\
    --logging_steps 100 \\
    --save_strategy epoch \\
    --evaluation_strategy epoch \\
    --project_name kt-dev-challenge \\
    --load_best_model_at_end
"""
        f.write(sh_template)

for sh in os.listdir(SH_LIST_DIR):
    path = os.path.join(SH_LIST_DIR, sh)
    os.system(f"sh {path}")
