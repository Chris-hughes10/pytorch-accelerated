.. pytorch-accelerated documentation master file

.. _timm: https://github.com/rwightman/pytorch-image-models
.. _transformers: https://huggingface.co/transformers/
.. _torchmetrics: https://torchmetrics.readthedocs.io/en/latest/
.. _accelerate: https://github.com/huggingface/accelerate
.. _examples: https://github.com/Chris-hughes10/pytorch-accelerated/tree/main/examples/

Welcome to `pytorch\-accelerated`'s documentation!
**************************************************

What is `pytorch-accelerated`?
================================

`pytorch\-accelerated` is a lightweight library designed to accelerate the process of training PyTorch models
by providing a minimal, but extensible training loop \- encapsulated in a single :class:`~pytorch_accelerated.trainer.Trainer`
object \- which is flexible enough to handle the majority of use cases, and capable of utilizing different hardware
options with no code changes required.

`pytorch\-accelerated` offers a streamlined feature set, and places a huge emphasis on **simplicity** and **transparency**,
to enable users to understand exactly what is going on under the hood, but without having to write and maintain the boilerplate themselves!

Key features
===============

- A simple and contained, but easily customisable, training loop, which should work out of the box in straightforward cases\; behaviour can be customised using inheritance and\/or callbacks.
- Handles device placement, mixed-precision, DeepSpeed integration, multi-GPU and distributed training with no code changes.
- Uses pure PyTorch components, with no additional modifications or wrappers, and easily interoperates with other popular libraries such as `timm`_, `transformers`_ and `torchmetrics`_.
- A small, streamlined API ensures that there is a minimal learning curve for existing PyTorch users.


Significant effort has been taken to ensure that every part of the library - both internal and external components - is as clear and simple as possible,
making it easy to customise, debug and understand exactly what is going on behind the scenes at each step\; most of the
behaviour of the trainer is contained in a single class!

In the spirit of Python, nothing is hidden and everything is accessible.

`pytorch\-accelerated` is proudly and transparently built on top of Hugging Face's `accelerate`_, which is responsible for the
movement of data between devices and launching of training configurations. When customizing the trainer, or launching
training, users are encouraged to consult the `Accelerate documentation <https://huggingface.co/docs/accelerate/>`_
to understand all available options; Accelerate provides convenient functions for operations such gathering tensors,
usage of which can be seen in the `pytorch\-accelerated` `examples`_ folder!

To learn more about the motivations behind this library, along with a detailed getting started guide, check out `this blog post <https://medium.com/@chris.p.hughes10/introducing-pytorch-accelerated-6ba99530608c?source=friends_link&sk=868c2d2ec5229fdea42877c0bf82b968>`_.


Who is pytorch-accelerated aimed at?
------------------------------------------

- Users that are familiar with PyTorch but would like to avoid having to write the common training loop boilerplate
  to focus on the interesting parts of the training loop.
- Users who like, and are comfortable with, selecting and creating their own models, loss functions, optimizers and datasets.
- Users who value a simple and streamlined feature set, where the behaviour is easy to debug, understand, and reason about!

Why shouldn't I use pytorch-accelerated?
---------------------------------------------

- If you are looking for an end-to-end solution, encompassing everything from loading data to inference,
  which helps you to select a model, optimizer or loss function, you would probably be better suited to
  `fastai <https://github.com/fastai/fastai>`_. `pytorch\-accelerated` focuses only on the training process, with all other
  concerns being left to the responsibility of the user.

- If you would like to write the entire training loop yourself, just without all of the device management headaches,
  you would probably be best suited to using `accelerate`_ directly! Whilst it is possible to customize every part
  of the :class:`~pytorch_accelerated.trainer.Trainer`, the training loop is fundamentally broken up into a number of
  different methods that you would have to override. But, before you go, is writing those *for* loops really important
  enough to warrant starting from scratch *again* 😉.

- If you are working on a custom, highly complex, use case which does not fit the patterns of usual training loops
  and want to squeeze out every last bit of performance on your chosen hardware, you are probably best off sticking
  with vanilla PyTorch; any high-level API becomes an overhead in highly specialized cases!

Acknowledgements
========================

Many aspects behind the design and features of `pytorch\-accelerated` were greatly inspired by a number of excellent
libraries and frameworks such as `fastai <https://github.com/fastai/fastai>`_, `timm`_,
`PyTorch-lightning <https://github.com/PyTorchLightning/pytorch-lightning>`_ and Hugging Face `accelerate`_. Each of these tools
have made an enormous impact on both this library and the machine learning community, and their influence can not be
stated enough!

`pytorch\-accelerated` has taken only inspiration from these tools, and all of the functionality contained has been implemented
from scratch in a way that benefits this library. The only exceptions to this are some of the scripts in the
`examples`_
folder in which existing resources were taken and modified in order to showcase the features of `pytorch\-accelerated`;
these cases are clearly marked, with acknowledgement being given to the original authors.

.. toctree::
   :maxdepth: 2
   :caption: Get Started:

   installation
   quickstart

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   trainer
   callbacks
   tracking
   run_config
   fine_tuning
   schedulers


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
