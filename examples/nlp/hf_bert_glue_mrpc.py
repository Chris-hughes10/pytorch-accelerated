########################################################################
#
# This example trains a Bert base model on GLUE MRPC and
# was adapted from an example produced by HuggingFace available here:
# https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
#
########################################################################


import argparse
from functools import partial

import torch
from datasets import load_dataset, load_metric
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from pytorch_accelerated.trainer import Trainer

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


class TransformersTrainer(Trainer):
    def __init__(self, model, optimizer, metric, *args, **kwargs):
        super().__init__(
            model=model, optimizer=optimizer, loss_func=None, *args, **kwargs
        )
        self.metric = metric

    def calculate_train_batch_loss(self, batch):
        outputs = self.model(**batch)

        return {
            "loss": outputs.loss,
            "model_outputs": outputs.logits,
            "batch_size": batch["attention_mask"].size(0),
        }

    def calculate_eval_batch_loss(self, batch):
        with torch.no_grad():
            outputs = self.model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        self.metric.add_batch(
            predictions=self._accelerator.gather(predictions),
            references=self._accelerator.gather(batch["labels"]),
        )
        return {
            "loss": outputs.loss,
            "model_outputs": outputs.logits,
            "batch_size": batch["attention_mask"].size(0),
        }

    def eval_epoch_end(self):
        self.run_history.update_metric("metrics", self.metric.compute())

    def create_scheduler(self):
        return self.create_scheduler_fn(
            optimizer=self.optimizer,
            num_training_steps=len(self._train_dataloader)
            * self.run_config["num_epochs"],
        )


def training_function(config, args):
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    correct_bias = config["correct_bias"]
    batch_size = int(config["batch_size"])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")
    metric = load_metric("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=None,
        )
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets.rename_column_("label", "labels")

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    # Instantiate the model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", return_dict=True
    )

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr, correct_bias=correct_bias)

    # Create an instance of our trainer
    trainer = TransformersTrainer(
        model=model, optimizer=optimizer, collate_fn=collate_fn, metric=metric
    )

    # Wrap the scheduler factory function as a higher order function so that it will be created inside the trainer
    lr_scheduler = partial(
        get_linear_schedule_with_warmup,
        num_warmup_steps=100,
    )

    # start training
    # Multiprocessing is handled by dataset, so override num workers
    trainer.train(
        num_epochs=num_epochs,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        per_device_batch_size=batch_size,
        train_dataloader_kwargs={"num_workers": 0},
        eval_dataloader_kwargs={"batch_size": EVAL_BATCH_SIZE, "num_workers": 0},
        create_scheduler_fn=lr_scheduler,
        gradient_accumulation_steps=gradient_accumulation_steps,
        collate_fn=collate_fn,
    )


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--fp16", action="store_true", help="If passed, will use FP16 training."
    )
    parser.add_argument(
        "--cpu", action="store_true", help="If passed, will train on the CPU."
    )
    args = parser.parse_args()
    config = {
        "lr": 2e-5,
        "num_epochs": 3,
        "correct_bias": True,
        "seed": 42,
        "batch_size": 16,
    }
    training_function(config, args)


if __name__ == "__main__":
    main()
