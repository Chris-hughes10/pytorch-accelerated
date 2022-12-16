FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime

# Need wget to download data
RUN apt-get update && apt-get install -y wget unzip

# Install pip dependencies
RUN pip install --no-cache-dir \
                'azureml-mlflow==1.39.0.post1' \
                'mlflow-skinny==1.26.1' \
                'func_to_script==0.1.0' \
                'albumentations==1.2.0' \
                'pandas==1.3.5' \
                'matplotlib==3.5.3' \
                'timm==0.6.11' \
                'torchmetrics==0.10.'

