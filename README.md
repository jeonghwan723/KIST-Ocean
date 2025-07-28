# KIST-Ocean
The Korea Institute of Science and Technology's Ocean model (KIST-Ocean) was developed based on a visual attention adversarial network composed of a generator and a discriminator (Guo et al., 2023; Li et al., 2023).

## Repository structure
> KIST-Ocean/: main directory
>> model/
>>> AVAN/: Python scripts for training KIST-Ocean model
>>>
>>>> train_v01.py: Python script for training KIST-OCean
>>>> 
>>>> config.py: Configuration for training and inference
>>>> 
>>>> AVAN_v01.py: Python script for the backbone of the KIST-Ocean model
>>>> 
>>>> utils.py: Python script for containing various utility functions
>>>> 
>>>> inferencer_GT.py: Python script for inferring the future ocean state by prescribing ground truth (observation) as the surface boundary condition (i.e., generating KIST-O_GT)
>>>
>>> output/: Directory where the trained model is saved

>> data/: Statistical datasets required for training and inference

## Requirements
- python v3.8.17
- torch v1.13.0
- timm v0.9.16
- netcdf4 v1.6.2
- numpy v1.24.3
- scipy v1.10.1
- matplotlib v3.7.5
- basemap v1.4.1

## Our Linux environment
- OS:CentOS Linux 7
- GPU: Nvidia A100
- CUDA version: 11.7

## References
- Guo, M.-H., Lu, C.-Z., Liu, Z.-N., Cheng, M.-M. & Hu, S.-M. Visual attention network. Comput. Vis. Media. 9, 733–752 (2023).
- Li, T., Yang, F. & Song, Y. Visual Attention Adversarial Networks for Chinese Font Translation. Electronics 12, 1388 (2023).

