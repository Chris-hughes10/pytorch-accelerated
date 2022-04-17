Quickstart
############

To get started, simply import and use the `pytorch\-accelerated` :class:`pytorch_accelerated.trainer.Trainer`, as demonstrated in the following snippet,
and then launch training using the `accelerate CLI <https://huggingface.co/docs/accelerate/quicktour.html#launching-your-distributed-script>`_ as described below::

    # examples/vision/train_mnist.py
    import os

    from torch import nn, optim
    from torch.utils.data import random_split
    from torchvision import transforms
    from torchvision.datasets import MNIST

    from pytorch_accelerated import Trainer

    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Sequential(
                nn.Linear(in_features=784, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=10),
            )

        def forward(self, input):
            return self.main(input.view(input.shape[0], -1))

    def main():
        dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
        train_dataset, validation_dataset, test_dataset = random_split(dataset, [50000, 5000, 5000])
        model = MNISTModel()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_func = nn.CrossEntropyLoss()

        trainer = Trainer(
                model,
                loss_func=loss_func,
                optimizer=optimizer,
        )

        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            num_epochs=8,
            per_device_batch_size=32,
        )

        trainer.evaluate(
            dataset=test_dataset,
            per_device_batch_size=64,
        )

    if __name__ == "__main__":
        main()

To launch training using the `accelerate CLI <https://huggingface.co/docs/accelerate/quicktour.html#launching-your-distributed-script>`_ on your machine(s), run::

    accelerate config --config_file accelerate_config.yaml

and answer the questions asked. This will generate a config file that will be used to properly set the default options when doing::

    accelerate launch --config_file accelerate_config.yaml train.py [--training-args]

.. Note::
    Using the `accelerate CLI <https://huggingface.co/docs/accelerate/quicktour.html#launching-your-distributed-script>`_ is completely optional,
    training can also be launched in the usual way using::

        python train.py / python -m torch.distributed ...

    depending on your infrastructure configuration, for users who would like to maintain a more fine-grained control
    over the launch command.

Running in a Notebook
=======================

`Accelerate <https://huggingface.co/docs/accelerate>`_ also provides a :func:`~accelerate.notebook_launcher` function, that can be used to launch distributed training from a notebook; which is especially useful for Colab or Kaggle notebooks.

To train a model using `pytorch_accelerated` from a notebook, just define the :func:`~pytorch_accelerated.trainer.Trainer` in a training_function,
and use this as an argument into :class:`~accelerate.notebook_launcher`. To run the example in above in a notebook, we would use::

    notebook_launcher(main, num_processes=num_gpus)

More information about training in a notebook can be `found here <https://huggingface.co/docs/accelerate/launcher.html>`_

Debugging with an IDE
=======================

Whilst ``pytorch_accelerated`` is primarily designed to be launched using the `accelerate CLI <https://huggingface.co/docs/accelerate/quicktour.html#launching-your-distributed-script>`_,
sometimes it's useful to debug a training script in your favourite editor to see exactly what's going on!

In these cases, we can simply use the :func:`~accelerate.notebook_launcher` function as described above. To debug the example above, after setting some breakpoints,
replace the lines::

    if __name__ == "__main__":
        main()

with::

    notebook_launcher(main, num_processes=num_gpus)

Next steps
=============

More complex training examples can be seen in the examples folder `here <https://github.com/Chris-hughes10/pytorch-accelerated/tree/main/examples/>`_

