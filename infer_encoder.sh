CUDA_DEVICE_ORDER=PCI_BUS_ID \ 
CUDA_VISIBLE_DEVICES=1 python infer_encoder.py \
--model_name_or_path ./encoder_only_models \
--test_file ./data/test1.csv \
--test_model ./models/encoder/0.0001_16_4_0.1/checkpoint-2082 \
--data_path ./data/klue_ner_test_20.txt
