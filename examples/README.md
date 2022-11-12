# Examples

These examples are designed to illustrate how PyTorch-accelerated can be used in different contexts.

Example | Location | Description | Dataset
:---:|:---:|:---:|:---:
Train MNIST (Vision classification)| [`core/train_mnist.py`](core/train_mnist.py) | Demonstrates how to train a custom model using the default Trainer | MNIST (included with TorchVision)
Integrate Metrics using a Callback| `core/train_with_metrics_in_callback.py` | Demonstrates how to integrate `TorchMetrics` metrics using a custom callback with the default Trainer| MNIST
Integrate Metrics using a Custom Trainer| `core/train_with_metrics_in_loop.py` | Demonstrates how to integrate  `TorchMetrics` metrics, using a custom Trainer | MNIST
Finetune BERT (NLP Sequence classification) | `nlp/hf_bert_glue_mrpc.py` | Finetunes a BERT model from `Transformers` using a custom Trainer | GLUE MRPC (included with `Datasets`)
Train Faster-RCNN (Vision object detection)| [`vision/faster_rcnn`](vision/faster_rcnn/train_cars.py) | Demonstrates how to create a custom Trainer for use with TorchVision's Faster-RCNN model | Cars (object detection)
Finetune a CNN (Vision classifier)| [`vision/transfer_learning/pets_finetune.py`](vision/transfer_learning/pets_finetune.py) | Demonstrates how to use gradual unfreezing when finetuning a model from `timm`| Pets
Finetune a CNN (Vision classifier)| [`vision/transfer_learning/pytorch_tutorial_finetune.py`](vision/transfer_learning/pytorch_tutorial_finetune.py) | Demonstrates how to use `TorchVision` models and datasets when finetuning a model| Ants and Bees
Progressive resizing (Vision classifier)| [`vision/transfer_learning/progressive_resizing.py`](vision/transfer_learning/progressive_resizing.py) | Demonstrates how to use progressive resizing, to gradually increase image size| Imagenette
Using all components from `timm` (Vision classifier)| [`vision/using_timm_components/all_timm_components.py`](vision/using_timm_components/all_timm_components.py) | Demonstrates how to create a custom `Trainer` to use Datasets, DataLoaders, models, optimizers and schedulers from `timm`| Imagenette
Using some components from `timm` (Vision classifier)| [`vision/using_timm_components/train_mixup_ema.py`](vision/using_timm_components/train_mixup_ema.py) | Demonstrates how to create a custom `Trainer` to use data augmentation and ModelEMA from `timm` alongside torch DataLoaders| Imagenette


## Running the examples

TODO: each example has its own dockerfile and devcontainer ...


## Datasets Used

TODO: provide download instructions, datasets should be downloaded to the data folder

- MNIST
- Imagenette
- Pets
- Cars
- GLUE MRPC