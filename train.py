import os

import wandb
import numpy as np
from datasets import load_metric
from dotenv import load_dotenv
from tqdm import tqdm

from transformers import (
    AutoConfig,
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    HfArgumentParser,
)

from args import (
    DataTrainingArguments,
    ModelArguments,
    NERTrainingArguments,
    LoggingArguments
)

from utils import (
    T5NERDataset,
    seed_everything
)


DATA_PATH = os.path.abspath("./data")
SUBMISSION_PATH = "./results"

device = "cuda:0"

METRIC = load_metric("./metrics/f1.py")


def main():
    parser = HfArgumentParser([DataTrainingArguments, ModelArguments, NERTrainingArguments, LoggingArguments])
    data_args, model_args, training_args, logging_args = parser.parse_args_into_dataclasses()

    seed_everything(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        batch_size = 32

        decoded_preds = []
        decoded_labels = []

        for batch in tqdm([preds[i: i + batch_size] for i in range(0, len(preds), batch_size)]):
            decoded_preds += tokenizer.batch_decode(batch, skip_special_tokens=True)
            
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        for batch in tqdm([labels[i: i + batch_size] for i in range(0, len(labels), batch_size)]):
            decoded_labels += tokenizer.batch_decode(batch, skip_special_tokens=True)

        result = METRIC.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    if training_args.do_train:
        train_dataset = T5NERDataset(
            file=os.path.join(DATA_PATH, data_args.train_file),
            tokenizer=tokenizer,
            max_len=data_args.max_length
        )

    if training_args.do_eval:
        valid_dataset = T5NERDataset(
            file=os.path.join(DATA_PATH, data_args.validation_file),
            tokenizer=tokenizer,
            max_len=data_args.max_length
        )

    if training_args.do_train:
        # -- Wandb
        load_dotenv(dotenv_path=logging_args.dotenv_path)
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)
        
        group_name = model_args.model_name_or_path
        name = f"EP:{training_args.num_train_epochs}"\
            f"_LR:{training_args.learning_rate}"\
            f"_BS:{training_args.per_device_train_batch_size}"\
            f"_WR:{training_args.warmup_ratio}"\
            f"_WD:{training_args.weight_decay}"
        
        wandb.init(
            entity="ku-nlp",
            project=logging_args.project_name,
            group=group_name,
            name=name
        )
        wandb.config.update(training_args)


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        # eval_examples=valid_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    best_path = os.path.join(training_args.output_dir, 'best_ckpt')
    os.makedirs(best_path, exist_ok=True)
    trainer.save_model(best_path)


if __name__ == "__main__":
    main()
