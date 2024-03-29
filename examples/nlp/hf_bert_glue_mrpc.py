# Modifications Copyright (C) 2021 Chris Hughes
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################
#
# This is an accelerated example which trains a Bert base model on GLUE MRPC and
# was adapted from an example produced by HuggingFace available here:
# https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py
#
# Note: This example requires installing the datasets, evaluate, and transformers packages
########################################################################


from functools import partial

import evaluate
import torch
from datasets import load_dataset
from func_to_script import script
from transformers import (
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
            predictions=self.gather(predictions),
            references=self.gather(batch["labels"]),
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
            num_training_steps=len(self._train_dataloader) * self.run_config.num_epochs,
        )


@script
def main(
    lr: float = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 16,
):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")
    metric = evaluate.load("glue", "mrpc")

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
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

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
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    # Create an instance of our trainer
    trainer = TransformersTrainer(model=model, optimizer=optimizer, metric=metric)

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


if __name__ == "__main__":
    main()
