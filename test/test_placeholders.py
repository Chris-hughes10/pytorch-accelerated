import math
from functools import partial

from pytest import mark

from pytorch_accelerated import TrainerPlaceholderValues
from pytorch_accelerated.run_config import TrainerRunConfig
from pytorch_accelerated.trainer import replace_trainer_placeholder_values


def create_run_config(
    train_dl_length,
    num_epochs=5,
    gradient_accumulation_steps=1,
    num_processes=1,
    train_batch_size=32,
    eval_batch_size=64,
):
    num_update_steps_per_epoch = math.ceil(
        train_dl_length / gradient_accumulation_steps
    )

    max_num_train_steps = num_epochs * num_update_steps_per_epoch

    return TrainerRunConfig(
        num_epochs=num_epochs,
        train_per_device_batch_size=train_batch_size,
        train_dl_kwargs={"tkwarg1": "tval1"},
        eval_per_device_batch_size=eval_batch_size,
        eval_dl_kwargs={"ekwarg1": "eval1"},
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip_value=None,
        train_total_batch_size=train_batch_size
        * num_processes
        * gradient_accumulation_steps,
        eval_total_batch_size=eval_batch_size * num_processes,
        max_num_train_steps=max_num_train_steps,
        is_local_process_zero=True,
        is_world_process_zero=True,
        is_distributed=True,
        mixed_precision="fp16",
        num_processes=1,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        num_local_update_steps_per_epoch=num_update_steps_per_epoch / num_processes,
    )


class FakeDataLoader:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class FakeTrainer:
    def __init__(
        self,
        train_dl_length,
        eval_dl_length,
        run_config,
    ):
        self._train_dataloader = FakeDataLoader(train_dl_length)
        self._eval_dataloader = FakeDataLoader(eval_dl_length)
        self.run_config = run_config


@mark.parametrize(
    (
        "expected_train_dl_length",
        "expected_eval_dl_length",
        "num_epochs",
        "num_processes",
        "gradient_accumulation_steps",
    ),
    [(1000, 500, 5, 1, 1), (800, 200, 10, 2, 2)],
)
def test_can_inject_placeholders(
    expected_train_dl_length,
    expected_eval_dl_length,
    num_epochs,
    num_processes,
    gradient_accumulation_steps,
):
    int_arg = 3
    str_arg = "test"

    trainer = FakeTrainer(
        expected_train_dl_length,
        expected_eval_dl_length,
        create_run_config(
            num_epochs=num_epochs,
            num_processes=num_processes,
            gradient_accumulation_steps=gradient_accumulation_steps,
            train_dl_length=expected_train_dl_length,
        ),
    )
    fn_with_placeholders = partial(
        dict,
        num_update_steps=TrainerPlaceholderValues.NUM_UPDATE_STEPS_PER_EPOCH,
        train_dl_length=TrainerPlaceholderValues.TRAIN_DATALOADER_LEN,
        eval_dl_length=TrainerPlaceholderValues.EVAL_DATALOADER_LEN,
        num_epochs=TrainerPlaceholderValues.NUM_EPOCHS,
        int_arg=int_arg,
        str_arg=str_arg,
    )

    replaced_fn = replace_trainer_placeholder_values(trainer, fn_with_placeholders)
    keyword_dict = replaced_fn()

    assert (
        keyword_dict["num_update_steps"]
        == trainer.run_config.num_update_steps_per_epoch
    )
    assert keyword_dict["num_epochs"] == num_epochs
    assert keyword_dict["train_dl_length"] == expected_train_dl_length
    assert keyword_dict["eval_dl_length"] == expected_eval_dl_length
    assert keyword_dict["int_arg"] == int_arg
    assert keyword_dict["str_arg"] == str_arg


def test_can_multiply_placeholder():
    expected_train_dl_length = 100
    expected_eval_dl_length = 50
    num_epochs = 5
    trainer = FakeTrainer(
        expected_train_dl_length,
        expected_eval_dl_length,
        create_run_config(
            num_epochs=num_epochs,
            train_dl_length=expected_train_dl_length,
        ),
    )
    fn_with_placeholders = partial(
        dict,
        num_epochs=TrainerPlaceholderValues.NUM_EPOCHS,
        num_epochs_modified=TrainerPlaceholderValues.NUM_EPOCHS * 2,
    )

    replaced_fn = replace_trainer_placeholder_values(trainer, fn_with_placeholders)
    keyword_dict = replaced_fn()

    assert keyword_dict["num_epochs"] == num_epochs
    assert keyword_dict["num_epochs_modified"] == 2 * num_epochs


def test_can_add_placeholder():
    expected_train_dl_length = 100
    expected_eval_dl_length = 50
    num_epochs = 5
    trainer = FakeTrainer(
        expected_train_dl_length,
        expected_eval_dl_length,
        create_run_config(
            num_epochs=num_epochs,
            train_dl_length=expected_train_dl_length,
        ),
    )
    fn_with_placeholders = partial(
        dict,
        num_epochs=TrainerPlaceholderValues.NUM_EPOCHS,
        num_epochs_modified=TrainerPlaceholderValues.NUM_EPOCHS + 2,
    )

    replaced_fn = replace_trainer_placeholder_values(trainer, fn_with_placeholders)
    keyword_dict = replaced_fn()

    assert keyword_dict["num_epochs"] == num_epochs
    assert keyword_dict["num_epochs_modified"] == 2 + num_epochs
