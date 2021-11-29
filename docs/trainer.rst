.. currentmodule:: pytorch_accelerated.trainer

.. _trainer:

Trainer
*********

.. autoclass:: Trainer

   .. automethod:: __init__

Training a model
==================

The main entrypoint for the :class:`Trainer` is the :meth:`~Trainer.train` method, which is used to launch a training run.

.. automethod:: Trainer.train

Using learning rate schedulers
---------------------------------

Note that, as the optimizer and dataloaders need to be internally prepared prior to training, in order to use a learning
rate scheduler, a factory function must be provided to :meth:`~Trainer.train` as the ``create_scheduler_fn`` argument.
This must be a function which accepts the optimizer as a single parameter and returns an instance of a learning rate scheduler.

.. note::
    Passing an instance of a PyTorch learning rate scheduler as the ``create_scheduler_fn`` argument
    to :meth:`~Trainer.train` will **not** work as intended.

A simple method of creating a scheduler factory function this is by using :meth:`functools.partial` like so::

    from functools import Partial

    from torch.optim import lr_scheduler

    create_scheduler_fn = partial(lr_scheduler.StepLR, step_size=7, gamma=0.1)


Using ``TrainerPlaceHolderValues``
--------------------------------------

.. autoclass:: TrainerPlaceholderValues

The list of the available placeholders are:

- ``NUM_EPOCHS``
- ``NUM_UPDATE_STEPS_PER_EPOCH``
- ``TRAIN_DATALOADER_LEN``
- ``EVAL_DATALOADER_LEN``

Alternatively, the same outcome could be achieved by overriding the :class:`Trainer`'s :meth:`~Trainer.create_scheduler`
method, which will be discussed below.


Evaluating a model
====================

Once a model has been trained, or loaded from a checkpoint, the :class:`~Trainer` can also be used for evaluation, which
consists of running a single epoch, using the :class:`Trainer`'s evaluation loop logic, on the given dataset.

.. automethod:: Trainer.evaluate

Utility Methods
==================

.. automethod:: Trainer.save_checkpoint
.. automethod:: Trainer.load_checkpoint
.. automethod:: pytorch_accelerated.trainer.Trainer.print

Customizing Trainer Behaviour
================================

Whilst the :class:`Trainer` should work out of the box in straightforward use cases, subclassing the trainer and overriding
its methods is intended and encouraged - think of the base implementation as a set of 'sensible defaults'!

.. Note::
    Methods which are prefixed with a verb such as *create* or *calculate* expect a value to be returned,
    all other methods are used to set internal state (e.g. ``optimizer.step()``)


Setup Methods
-------------

.. automethod:: Trainer.create_train_dataloader
.. automethod:: Trainer.get_default_train_dl_kwargs
.. automethod:: Trainer.create_eval_dataloader
.. automethod:: Trainer.get_default_eval_dl_kwargs
.. automethod:: Trainer.create_scheduler

Training Run Methods
---------------------------------

.. automethod:: Trainer.training_run_start
.. automethod:: Trainer.training_run_epoch_end
.. automethod:: Trainer.training_run_end


Training epoch Methods
++++++++++++++++++++++++

.. automethod:: Trainer.train_epoch_start
.. automethod:: Trainer.calculate_train_batch_loss
.. automethod:: Trainer.backward_step
.. automethod:: Trainer.optimizer_step
.. automethod:: Trainer.scheduler_step
.. automethod:: Trainer.optimizer_zero_grad
.. automethod:: Trainer.train_epoch_end

Evaluation epoch Methods
++++++++++++++++++++++++++

.. automethod:: Trainer.eval_epoch_start
.. automethod:: Trainer.calculate_eval_batch_loss
.. automethod:: Trainer.eval_epoch_end

Evaluation Run Methods
---------------------------------

.. automethod:: Trainer.evaluation_run_start
.. automethod:: Trainer.evaluation_run_end


Internal Methods
--------------------

.. Warning::
    In the spirit of Python, nothing is truly hidden within the :class:`Trainer`. However, care must be taken as, by
    overriding these methods, you are fundamentally changing how the :class:`Trainer` is working internally and this may have
    untended consequences. When modifying one or more internal methods, it is the responsibility of the user to ensure that
    the :class:`Trainer` continues to work as intended!

Internal Setup
++++++++++++++++++

.. automethod:: Trainer._prepare_model_and_optimizer
.. automethod:: Trainer._prepare_dataloaders
.. automethod:: Trainer._create_run_config

Training run behaviour
++++++++++++++++++++++++

.. automethod:: Trainer._run_training

Training epoch behaviour
++++++++++++++++++++++++++

.. automethod:: Trainer._run_train_epoch
.. automethod:: Trainer._clip_gradients

Evaluation epoch behaviour
++++++++++++++++++++++++++

.. automethod:: Trainer._run_eval_epoch


Should I subclass the Trainer or use a callback?
---------------------------------------------------

The behaviour of the :class:`Trainer` can also be extended using Callbacks. All callback methods are prefixed with ``on_``.

It is recommended that callbacks are used to contain 'infrastructure' code, which is not essential to the operation of the training loop,
such as logging, but this decision is left to the judgement of the user based on the specific use case. If it seems
overkill to subclass the :class:`Trainer` for the modification you wish to make, it may be better to use a callback instead.

For more information on callbacks, see :ref:`callbacks`.

.. _trainer_metric_example:

Recording metrics
===================

The :class:`Trainer` contains an instance of :class:`~pytorch_accelerated.tracking.RunHistory`, which can be used to store and retrieve the values of
any metrics to track during training. By default, the only metrics that are recorded by the :class:`Trainer` are
the losses observed during training and evaluation.

.. Note::
    If the callback :class:`~pytorch_accelerated.callbacks.PrintMetricsCallback` is being used, any metrics recorded
    in the run history will be printed to the console automatically.

The API for :class:`~pytorch_accelerated.tracking.RunHistory` is detailed at :ref:`run_history`.

Here is an example of how we can subclass the :class:`~trainer.Trainer` and use the :class:`~pytorch_accelerated.tracking.RunHistory` to track metrics
computed using `TorchMetrics <https://torchmetrics.readthedocs.io/en/latest/pages/overview.html>`_::

    from torchmetrics import MetricCollection, Accuracy, Precision, Recall
    from pytorch_accelerated import Trainer

    class TrainerWithMetrics(Trainer):
        def __init__(self, num_classes, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # this will be moved to the correct device automatically by the
            # MoveModulesToDeviceCallback callback, which is used by default
            self.metrics = MetricCollection(
                {
                    "accuracy": Accuracy(num_classes=num_classes),
                    "precision": Precision(num_classes=num_classes),
                    "recall": Recall(num_classes=num_classes),
                }
            )

        def calculate_eval_batch_loss(self, batch):
            batch_output = super().calculate_eval_batch_loss(batch)
            preds = batch_output["model_outputs"].argmax(dim=-1)

            self.metrics.update(preds, batch[1])

            return batch_output

        def eval_epoch_end(self):
            metrics = self.metrics.compute()
            self.run_history.update_metric("accuracy", metrics["accuracy"].cpu())
            self.run_history.update_metric("precision", metrics["precision"].cpu())
            self.run_history.update_metric("recall", metrics["recall"].cpu())

            self.metrics.reset()


.. Note::
    If you feel that subclassing the :class:`Trainer` seems too excessive for this use case, this could also be done using a callback
    as demonstrated in :ref:`callbacks_metric_example`.

.. _inside-trainer:

What goes on inside the Trainer?
==================================

In pseudocode, the execution of a training run can be depicted as::

    train_dl = create_train_dataloader()
    eval_dl = create_eval_dataloader()
    scheduler = create_scheduler()

    training_run_start()
    on_training_run_start()

    for epoch in num_epochs:
        train_epoch_start()
        on_train_epoch_start()
        for batch in train_dl:
            on_train_step_start()
            batch_output = calculate_train_batch_loss(batch)
            on_train_step_end(batch, batch_output)
            backward_step(batch_output["loss"])
            optimizer_step()
            scheduler_step()
            optimizer_zero_grad()
        train_epoch_end()
        on_train_epoch_end()

        eval_epoch_start()
        on_eval_epoch_start()
        for batch in eval_dl:
            on_eval_step_start()
            batch_output = calculate_eval_batch_loss(batch)
            on_eval_step_end(batch, batch_output)
        eval_epoch_end()
        on_eval_epoch_end()

        training_run_epoch_end()
        on_training_run_epoch_end()

    training_run_end()
    on_training_run_end()


Similarly, the execution of an evaluation run can be depicted as::

    eval_dl = create_eval_dataloader()

    evaluation_run_start()
    on_evaluation_run_start()

    eval_epoch_start()
    on_eval_epoch_start()
    for batch in eval_dl:
        on_eval_step_start()
        batch_output = calculate_eval_batch_loss(batch)
        on_eval_step_end(batch, batch_output)
    eval_epoch_end()
    on_eval_epoch_end()

    evaluation_run_end()
    on_evaluation_run_end()


The best way to understand how the :class:`Trainer` works internally is by examining the source code for the :meth:`~Trainer.train` method;
significant care has gone into making the internal methods as clean and clear as possible.