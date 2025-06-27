# torch_trt_utils
some tools package for torch_trt

# Environment

```bash
# apt install nvidia-driver-535

# install CUDA-12.4 + cuDNN 9.10.2
## omitted how to install CUDA-12.4, here
cat >> ~/.bashrc <<EOF
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
# nvcc -V to check CUDA version

wget https://developer.download.nvidia.com/compute/cudnn/9.10.2/local_installers/cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
dpkg -i cudnn-local-repo-ubuntu2204-9.10.2_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2204-9.10.2/cudnn-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudnn-cuda-12

# install TensorRT
###################################################### TensorRT Backend #######################################################
# CPP version need envs below
export TENSORRT_HOME=/usr/local/TensorRT/TensorRT-10.4.0.26
export LD_LIBRARY_PATH=$PATH:$TENSORRT_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PATH:$TENSORRT_HOME/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$TENSORRT_HOME/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$TENSORRT_HOME/include
###############################################################################################################################
```