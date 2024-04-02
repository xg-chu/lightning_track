FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# setting
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive
ENV ZSH_DISABLE_COMPFIX=true
ENV ZDOTDIR=/home/mil/x-chu/.zsh_docker

# install 
RUN apt-get update && apt-get install -y apt-utils wget vim git zsh nodejs ffmpeg libsm6 libxext6 sudo tmux \
                   && apt-get clean \
                   && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir iopath==0.1.9 fvcore==0.1.5.post20221221
RUN pip install --no-cache-dir pytorch3d==0.7.5 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html
RUN pip install --no-cache-dir mediapipe==0.10.7 tqdm==4.66.1 rich==13.7.0 lmdb==1.4.1 colored==2.2.4 einops==0.7.0 ninja==1.11.1.1 av==11.0.0 gpustat==1.1.1
RUN pip install --no-cache-dir opencv-python==4.9.0.80 scikit-image==0.22.0 onnxruntime-gpu==1.16.3 onnx==1.15.0 transformers==4.36.2
RUN pip install --no-cache-dir lightning==2.1.3 tensorboardX==2.6.2.2 ipdb==0.13.13 omegaconf==2.3.0 pykalman==0.9.7

WORKDIR /lightning_track
ADD . .
