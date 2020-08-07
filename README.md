# Lung Segmentation
Implemented Image Segmentation pipeline using 2 models [U-Net](https://arxiv.org/abs/1505.04597) and [ResU-Net](https://arxiv.org/pdf/1711.10684.pdf) and borrowed idea of [Multi Input and DeepSupervion](https://arxiv.org/abs/1810.07842) 


## DataSet
Used [Montgemory](https://lhncbc.nlm.nih.gov/publication/pub9931) dataset for training(70%) and validation(30%)  


## Models
* U-Net
* ResU-Net
* U-Net with Input Image pyramid and deep supervision
* ResU-Net with Input Image pyramid and deep supervision

## Implementation
Implemented in PyTorch   

Pre procossising, data augmentation and data loading is done in 'dataloader_cxr.py'. Training code in 'train.py'.Testing and output segmentation mask  generation in 'test.py'.   Mode 

**Model Code**
* Unet in 'cxr_resunet.py'
* ResUNet 'cxr_resunet.py'
* UNet with Input Image pyramid and deep supervision 'cxr_multiinput_resunet.py'
* ResUNet with Input Image pyramid and deep supervision 'cxr_multiinput_resunet.py'

**Loss**
Dice loss and dice loss with deep supervision in 'losses.py'

## Results(Dice Score)

|      | Without MIDS  | With MIDS |
|:----:|:-----:|:----------:|
|  U-Net | 0.918 |    0.947   |
| ResU-Net | 0.959 |    0.970   |

*MIDS= Multi Input Deep Super Vision*


![](https://github.com/Mustafa-Ali-iitb/Lung-Segmentation/blob/master/Final_Result.png?raw=true)
