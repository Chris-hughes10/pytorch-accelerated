# Examples

These examples are designed to illustrate how PyTorch-accelerated can be used in different contexts.

Example | Location | Description | Dataset
:---:|:---:|:---:|:---:
Train MNIST (Vision classification)| [`core/train_mnist.py`](core/train_mnist.py) | Demonstrates how to train a custom model using the default Trainer | MNIST (included with TorchVision)
Integrate Metrics using a Callback| [`core/train_with_metrics_in_callback.py`](core/train_with_metrics_in_callback.py) | Demonstrates how to integrate `TorchMetrics` metrics using a custom callback with the default Trainer| MNIST (included with TorchVision)
Integrate Metrics using a Custom Trainer| [`core/train_with_metrics_in_loop.py`](core/train_with_metrics_in_loop.py) | Demonstrates how to integrate  `TorchMetrics` metrics, using a custom Trainer | MNIST (included with TorchVision)
Finetune BERT (NLP Sequence classification) | [`nlp/hf_bert_glue_mrpc.py`](nlp/hf_bert_glue_mrpc.py) | Finetunes a BERT model from `Transformers` using a custom Trainer | GLUE MRPC (included with `Datasets`)
Train Faster-RCNN (Vision object detection)| [`vision/faster_rcnn/train_cars.py`](vision/faster_rcnn/train_cars.py) | Demonstrates how to create a custom Trainer for use with TorchVision's Faster-RCNN model | Cars (object detection)
Finetune a CNN (Vision classifier)| [`vision/transfer_learning/pets_finetune.py`](vision/transfer_learning/pets_finetune.py) | Demonstrates how to use gradual unfreezing when finetuning a model from `timm`| Pets
Finetune a CNN (Vision classifier)| [`vision/transfer_learning/pytorch_tutorial_finetune.py`](vision/transfer_learning/pytorch_tutorial_finetune.py) | Demonstrates how to use `TorchVision` models and datasets when finetuning a model| Ants and Bees
Progressive resizing (Vision classifier)| [`vision/transfer_learning/progressive_resizing.py`](vision/transfer_learning/progressive_resizing.py) | Demonstrates how to use progressive resizing, to gradually increase image size| Imagenette
Using all components from `timm` (Vision classifier)| [`vision/using_timm_components/all_timm_components.py`](vision/using_timm_components/all_timm_components.py) | Demonstrates how to create a custom `Trainer` to use Datasets, DataLoaders, models, optimizers and schedulers from `timm`| Imagenette
Using some components from `timm` (Vision classifier)| [`vision/using_timm_components/train_mixup_ema.py`](vision/using_timm_components/train_mixup_ema.py) | Demonstrates how to create a custom `Trainer` to use data augmentation and ModelEMA from `timm` alongside torch DataLoaders| Imagenette


## Running the examples

Each example has an associated `DockerFile` and Dev Container definition, which is located at the root of the folder, these define the execution environment for the example.; some examples share execution environments.

The easiest way to run an example is to open the relevant example in a dev container using [Visual Studio Code](https://code.visualstudio.com/), as described [here](https://code.visualstudio.com/docs/devcontainers/containers).


## Datasets Used

Some examples require external datasets to be downloaded; it is expected that all datasets are located in the `data` folder. Details of the datasets used are included below:

- MNIST: A vision dataset included with `TorchVision`, this will be downloaded when the example script is executed.
- GLUE MRPC: A NLP dataset included with [`datasets`](https://huggingface.co/docs/datasets/index), this will be downloaded when the example script is executed.
- Ants & Bees: A vision dataset which must be manually downloaded before use, a helper script is located in the `data` folder.
- Imagenette: A vision dataset which must be manually downloaded before use, a helper script is located in the `data` folder.
- Pets: A vision dataset which must be manually downloaded before use, a helper script is located in the `data` folder.
- Cars: An object detection dataset, which can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/sshikamaru/car-object-detection)
