FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# set bash as current shell
RUN apt-get update
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda
RUN apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion gcc g++ nano && \
        apt-get clean
    
WORKDIR /pai
        
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID pai_user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID pai_user
RUN apt-get -y install sudo
RUN echo "pai_user:pai_user" | chpasswd && adduser pai_user sudo

RUN echo "pai_user     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER pai_user

RUN sudo chown pai_user /pai && sudo chgrp pai_user /pai


RUN sudo apt-get install -y wget
WORKDIR /home/pai_user
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN sudo bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin
RUN sudo rm -f Miniconda3-latest-Linux-x86_64.sh  
RUN mkdir /home/pai_user/.conda/
RUN sudo chmod 777 /home/pai_user/.conda/


RUN conda init bash

# Create conda environment
RUN conda init
RUN conda create -n conda_env python=3.9 -y

RUN sh -c "echo set -o ignoreeof >> ~/.bashrc"
RUN sh -c "echo \"conda init\" >> ~/.bashrc"
RUN sh -c "echo \"conda activate conda_env\" >> ~/.bashrc"
