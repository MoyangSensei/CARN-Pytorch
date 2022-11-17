# CARN-PyTorch

This repository is an official PyTorch implementation of the paper **"A Super-Resolution Network Using Channel Attention Retention for Pathology Images"**.

![](https://github.com/MoyangSensei/CARN-Pytorch/blob/main/fig/11.png)

## Requirement
* torch == 1.8.0
* python >=3.6.0
* cuda == 11.1
* opencv-python == 3.4.1.15
* Pillow == 5.4.1
* numpy == 1.19.5

See `requirement.txt` for other details of the operating environment. 

Note that for `opencv-python`, this code has only tried the version provided above. Using an excessively high version may cause errors.

## Dataset: bcSR

We have provided 3 urls for bcSR:

`BaiduYun`:

https://pan.baidu.com/s/1gP9TNLh2Drw-Duz_KL5-Hw  
password: m1qt

`Google Drive`:

https://drive.google.com/drive/folders/1RDbQePqbMQzEY7Pc477fwRUNBOIJbuXp?usp=sharing

`figshare`:

https://figshare.com/articles/figure/bcSR/21155584

## How to train CARN?

First, prepare the code and dataset. The download method of the data set has been provided. To place the code folder and data set in any path, you need to unzip the dataset.

Next, modify the parameters related to the dataset in `option.py`, specifically the location where you place the dataset and the name of the dataset. 

You can find the commmand in `run.sh` that the benchmark result used.

## How to test CARN?

See the commmand in `run.sh`. In fact, you only need to add some parameters to the training command.

## Benchmark results in paper

![](https://github.com/MoyangSensei/CARN-Pytorch/blob/main/fig/22.png)

![](https://github.com/MoyangSensei/CARN-Pytorch/blob/main/fig/33.png)

## Information of comparison model

Some information about the comparison model used in the paper, including the source paper of the model, GitHub address and other conclusions.

### bicubic

This can be easily found on the Internet.

### SRCNN

* paper: Accelerating the super-resolution convolutional neural network
* code: https://github.com/yjn870/SRCNN-pytorch

This code cannot run correctly on our device, and it does not provide 8x sampling schemes. We have modified some details on this basis.

### SRGAN

* paper: Photo-realistic single image super-resolution using a generative adversarial network
* code: https://github.com/tensorlayer/srgan

This code cannot run correctly on our device, and it does not provide 8x sampling schemes. We have modified some details on this basis.

### EDSR

* paper: Enhanced deep residual networks for single image super-resolution
* code: https://github.com/sanghyun-son/EDSR-PyTorch

This code provides the implementation of EDSR, RDN and RCAN. The following models directly follow the content of this code.

### RDN

* paper: Residual dense network for image super-resolution
* code: https://github.com/sanghyun-son/EDSR-PyTorch

### RCAN

* paper: Image super-resolution using very deep residual channel attention networks
* code: https://github.com/sanghyun-son/EDSR-PyTorch

### SWD-Net

* paper: Joint spatial-wavelet dual-stream network for super-resolution
* code: https://github.com/franciszchen/SWD-Net

This code does not provide a scheme to use other data for training, and all multiples except 2x upsampling. We have asked questions in this code, and the author has now announced the method of making training data sets. We rewrite the feature upsampling part of this code in the way of CARN to realize the 3x, 4x and 8x upsampling tasks.

*****

# Work Log

* 2022.5.26
  * First upload this work. 

* 2022.7.17
  * Added the overall architecture diagram of CARN;
  * Updated `Requirement`;
  * Updated `dataset: MCSR`: provides the download method of Google Drive;
  * Updated `How to train CARN?`: provide more details for the training of this code;
  * Updated `How to test CARN?`: provide more details for the testing of this code;
  * Updated `Benchmark results in paper`: show the results in the paper;
  * Updated `Information of comparison model`: provide information about the comparison model used in the paper.

* 2022.8.25
  * Modify the dataset name: from `mcSR` to `bcSR`; 
  * Updated images of benchmark results.

* 2022.11.17
  * Updated all codes, added necessary comments, standardized functions; 
  * Provides a new way to obtain bcSR: `figshare`;
  * Updated benchmark results for quantitative evaluation;
  * Adjusted the layout of the network structure diagram.
