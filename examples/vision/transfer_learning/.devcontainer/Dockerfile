FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Need wget to download data
RUN apt-get update && apt-get install -y wget unzip

# Install pip dependencies
RUN pip install --no-cache-dir \
                'azureml-mlflow==1.39.0.post1' \
                'mlflow-skinny==1.26.1' \
                'func_to_script==0.1.0' \
                'pandas==1.3.5' \
                'matplotlib==3.5.3' \
                'timm==0.6.12' \
                'torchmetrics==0.11.4'

