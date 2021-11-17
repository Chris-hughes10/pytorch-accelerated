.. currentmodule:: callbacks

.. _callbacks:

Callbacks
*********

In addition to overridable hooks, the :class:`~trainer.Trainer` also includes a callback system.

It is recommended that callbacks are used to contain 'infrastructure' code, which is not essential to the operation of
the training loop, such as logging, but this decision is left to the judgement of the user based on the specific use case.

.. warning::
    Callbacks are executed sequentially, so if a callback is used to modify state, such as updating a metric, it is the
    responsibility of the user to ensure that this callback is placed before any callback which will read this state
    (i.e. for logging purposes)!

.. note::
    Callbacks are called **after** their corresponding hooks, e.g., a callback's ``on_train_epoch_end`` method is called
    *after* the method :meth:`trainer.Trainer.train_epoch_end`. This is done to support the pattern of updating the
    trainer's state in a method before reading this state in a callback.

    For more info on execution order within the training loop, see: :ref:`inside-trainer`.

Implemented Callbacks
======================

.. autoclass:: callbacks.TerminateOnNaNCallback
    :show-inheritance:

.. autoclass:: callbacks.PrintMetricsCallback
    :show-inheritance:

.. autoclass:: callbacks.PrintProgressCallback
    :show-inheritance:

.. autoclass:: callbacks.ProgressBarCallback
    :show-inheritance:

.. autoclass:: callbacks.SaveBestModelCallback
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: callbacks.EarlyStoppingCallback
    :show-inheritance:

    .. automethod:: __init__

Creating New Callbacks
========================

To create a new callback containing custom behaviour, e.g. logging to an external platform, it is recommended to subclass
:class:`~callbacks.TrainerCallback`. To avoid confusion with the :class:`~trainer.Trainer`'s methods, all callback methods are
prefixed with ``_on``.

.. warning::
    For maximum flexibility, the current instance of the :class:`~trainer.Trainer` is available in every callback method.
    However, changing the trainer state within a callback can have unintended consequences, as this may affect other parts
    of the training run. If a callback is used to modify :class:`~trainer.Trainer` state, it is responsibility of the user
    to ensure that everything continues to work as intended.

.. autoclass:: callbacks.TrainerCallback

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

A training run may be stopped early by raising a :class:`~callbacks.StopTrainingError`

.. _callbacks_metric_example:

Example: Tracking metrics using a callback
---------------------------------------------

By default, the only metrics that are recorded by the :class:`trainer.Trainer` are the losses observed during
training and evaluation. To track additional metrics, we can extend this behaviour using a callback.

Here is an example of how we can define a callback and use the :class:`~tracking.RunHistory` to track metrics
computed using `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/pages/overview.html>`_::

    class ClassificationMetricsCallback(TrainerCallback):
    def __init__(self, num_classes):
        self.cm_metrics = ConfusionMatrix(num_classes=num_classes)
        self.accuracy = Accuracy(num_classes=num_classes)

    def _move_to_device(self, trainer):
        self.cm_metrics.to(trainer._eval_dataloader.device)
        self.accuracy.to(trainer._eval_dataloader.device)

    def on_training_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_evaluation_run_start(self, trainer, **kwargs):
        self._move_to_device(trainer)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.cm_metrics.update(preds, batch[1])
        self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric(
            "confusion_matrix", self.cm_metrics.compute().cpu()
        )
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())

        self.cm_metrics.reset()
        self.accuracy.reset()

.. Note::
    If you feel that it would be clearer to compute metrics as part of the training loop, this could also be done by
    subclassing the :class:`trainer.Trainer` as demonstrated in :ref:`trainer_metric_example`.

Callback handler
======================

The execution of any callbacks passed to the :class:`~trainer.Trainer` is handled by an instance of an internal
callback handler class.


.. autoclass:: callbacks.CallbackHandler
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
