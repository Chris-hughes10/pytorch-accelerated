.. currentmodule:: pytorch_accelerated.callbacks

.. _callbacks:

Callbacks
*********

In addition to overridable hooks, the :class:`~pytorch_accelerated.trainer.Trainer` also includes a callback system.

It is recommended that callbacks are used to contain 'infrastructure' code, which is not essential to the operation of
the training loop, such as logging, but this decision is left to the judgement of the user based on the specific use case.

.. warning::
    Callbacks are executed sequentially, so if a callback is used to modify state, such as updating a metric, it is the
    responsibility of the user to ensure that this callback is placed before any callback which will read this state
    (i.e. for logging purposes)!

.. note::
    Callbacks are called **after** their corresponding hooks, e.g., a callback's ``on_train_epoch_end`` method is called
    *after* the method :meth:`pytorch_accelerated.trainer.Trainer.train_epoch_end`. This is done to support the pattern of updating the
    trainer's state in a method before reading this state in a callback.

    For more info on execution order within the training loop, see: :ref:`inside-trainer`.

Implemented Callbacks
======================

.. autoclass:: TerminateOnNaNCallback
    :show-inheritance:

.. autoclass:: LogMetricsCallback
    :show-inheritance:

.. autoclass:: PrintProgressCallback
    :show-inheritance:

.. autoclass:: ProgressBarCallback
    :show-inheritance:

.. autoclass:: SaveBestModelCallback
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: EarlyStoppingCallback
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: MoveModulesToDeviceCallback
    :show-inheritance:


Creating New Callbacks
========================

To create a new callback containing custom behaviour, e.g. logging to an external platform, it is recommended to subclass
:class:`~TrainerCallback`. To avoid confusion with the :class:`~pytorch_accelerated.trainer.Trainer`'s methods, all callback methods are
prefixed with ``_on``.

.. warning::
    For maximum flexibility, the current instance of the :class:`~pytorch_accelerated.trainer.Trainer` is available in every callback method.
    However, changing the trainer state within a callback can have unintended consequences, as this may affect other parts
    of the training run. If a callback is used to modify :class:`~pytorch_accelerated.trainer.Trainer` state, it is responsibility of the user
    to ensure that everything continues to work as intended.

.. autoclass:: TrainerCallback

    .. automethod:: TrainerCallback.on_init_end
    .. automethod:: TrainerCallback.on_training_run_start
    .. automethod:: TrainerCallback.on_train_epoch_start
    .. automethod:: TrainerCallback.on_train_step_start
    .. automethod:: TrainerCallback.on_train_step_end
    .. automethod:: TrainerCallback.on_train_epoch_end
    .. automethod:: TrainerCallback.on_eval_epoch_start
    .. automethod:: TrainerCallback.on_eval_step_start
    .. automethod:: TrainerCallback.on_eval_step_end
    .. automethod:: TrainerCallback.on_eval_epoch_end
    .. automethod:: TrainerCallback.on_training_run_end
    .. automethod:: TrainerCallback.on_stop_training_error

Stopping Training Early
--------------------------

A training run may be stopped early by raising a :class:`~StopTrainingError`

.. _callbacks_metric_example:

Example: Tracking metrics using a callback
---------------------------------------------

By default, the only metrics that are recorded by the :class:`pytorch_accelerated.trainer.Trainer` are the losses observed during
training and evaluation. To track additional metrics, we can extend this behaviour using a callback.

Here is an example of how we can define a callback and use the :class:`~pytorch_accelerated.tracking.RunHistory` to track metrics
computed using `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/pages/overview.html>`_::

    from torchmetrics import MetricCollection, Accuracy, Precision, Recall

    class ClassificationMetricsCallback(TrainerCallback):
        def __init__(self, num_classes):
            self.metrics = MetricCollection(
                {
                    "accuracy": Accuracy(num_classes=num_classes),
                    "precision": Precision(num_classes=num_classes),
                    "recall": Recall(num_classes=num_classes),
                }
            )

        def _move_to_device(self, trainer):
            self.metrics.to(trainer.device)

        def on_training_run_start(self, trainer, **kwargs):
            self._move_to_device(trainer)

        def on_evaluation_run_start(self, trainer, **kwargs):
            self._move_to_device(trainer)

        def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
            preds = batch_output["model_outputs"].argmax(dim=-1)
            self.metrics.update(preds, batch[1])

        def on_eval_epoch_end(self, trainer, **kwargs):
            metrics = self.metrics.compute()
            trainer.run_history.update_metric("accuracy", metrics["accuracy"].cpu())
            trainer.run_history.update_metric("precision", metrics["precision"].cpu())
            trainer.run_history.update_metric("recall", metrics["recall"].cpu())

            self.metrics.reset()

.. Note::
    If you feel that it would be clearer to compute metrics as part of the training loop, this could also be done by
    subclassing the :class:`pytorch_accelerated.trainer.Trainer` as demonstrated in :ref:`trainer_metric_example`.

Example: Create a custom logging callback
---------------------------------------------

It is recommended that callbacks are used to handle logging, to keep the training loop focused on the ML related code.
It is easy to create loggers for other platforms by subclassing the :class:`LogMetricsCallback` callback. For example,
we can create a logger for AzureML (which uses the MLFlow API) as demonstrated below::

    import mlflow

    class AzureMLLoggerCallback(LogMetricsCallback):
        def __init__(self):
            mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

        def on_training_run_start(self, trainer, **kwargs):
            mlflow.set_tags(trainer.run_config.to_dict())

        def log_metrics(self, trainer, metrics):
            if trainer.run_config.is_world_process_zero:
                mlflow.log_metrics(metrics)
            
Example: Create a custom callback to save predictions on evaluation
-------------------------------------------------------------------

Here is an example custom callback to record predictions during evaluation and then save them to csv at the end of the evaluation epoch. 

    from collections import defaultdict
    import pandas as pd

    class SavePredictionsCallback(TrainerCallback):

        def __init__(self, out_filename='./outputs/valid_predictions.csv') -> None:
            super().__init__()
            self.predictions = defaultdict(list)
            self.out_filename = out_filename

        def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
            input_features, targets = batch
            class_preds = trainer.gather(batch_output['model_outputs']).argmax(dim=-1)
            self.predictions['prediction'].extend(class_preds.cpu().tolist())
            self.predictions['targets'].extend(targets.cpu().tolist())

        def on_eval_epoch_end(self, trainer, **kwargs):
            trainer._accelerator.wait_for_everyone()
            if trainer.run_config.is_local_process_zero:
                df = pd.DataFrame.from_dict(self.predictions)
                df.to_csv(f'{self.out_filename}', index=False)
            

Callback handler
======================

The execution of any callbacks passed to the :class:`~pytorch_accelerated.trainer.Trainer` is handled by an instance of an internal
callback handler class.


.. autoclass:: CallbackHandler
   :members:

Creating new callback events
-------------------------------

To add even more flexibility, it is relatively simple to define custom callback events, and use them in the training loop::

    class VerifyBatchCallback(TrainerCallback):
        def verify_train_batch(self, trainer, xb, yb):
            assert xb.shape[0] == trainer.run_config["train_per_device_batch_size"]
            assert xb.shape[1] == 1
            assert xb.shape[2] == 28
            assert xb.shape[3] == 28
            assert yb.shape[0] == trainer.run_config["train_per_device_batch_size"]


    class TrainerWithCustomCallbackEvent(Trainer):
        def calculate_train_batch_loss(self, batch) -> dict:
            xb, yb = batch
            self.callback_handler.call_event(
                "verify_train_batch", trainer=self, xb=xb, yb=yb
            )
            return super().calculate_train_batch_loss(batch)
