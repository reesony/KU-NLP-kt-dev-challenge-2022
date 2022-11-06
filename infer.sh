CUDA_VISIBLE_DEVICES=0 python3 infer.py \
--model_name_or_path ./models/base_model \
--test_file ./data/test.csv \
--num_beams 5 \
--batch_size 128
