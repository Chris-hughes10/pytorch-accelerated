# Examples

These examples are designed to illustrate how PyTorch-accelerated can be used in different contexts.

Example | Location | Description | Dataset
:---:|:---:|:---:|:---:
Train MNIST (Vision classification)| [`core/train_mnist.py`](core/train_mnist.py) | Demonstrates how to train a custom model using the default Trainer | MNIST (included with TorchVision)
Integrate Metrics using a Callback| `metrics/train_with_metrics_in_callback.py` | Demonstrates how to integrate `TorchMetrics` metrics using a custom callback with the default Trainer| MNIST
Integrate Metrics using a Custom Trainer| `metrics/train_with_metrics_in_loop.py` | Demonstrates how to integrate  `TorchMetrics` metrics, using a custom Trainer | MNIST
Finetune BERT (NLP Sequence classification) | `nlp/hf_bert_glue_mrpc.py` | Finetunes a BERT model from `Transformers` using a custom Trainer | GLUE MRPC (included with `Datasets`)
Train Faster-RCNN (Vision object detection)| [`vision/faster_rcnn`](vision/faster_rcnn/train_cars.py) | Demonstrates how to create a custom Trainer for use with TorchVision's Faster-RCNN model | Cars (object detection)


## Running the examples

TODO: each example has its own dockerfile and devcontainer ...


## Datasets Used

TODO: provide download instructions, datasets should be downloaded to the data folder

- MNIST
- Imagenette
- Pets
- Cars
- GLUE MRPC