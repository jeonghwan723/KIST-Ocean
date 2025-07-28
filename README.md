# KIST-Ocean
The Korea Institute of Science and Technology's ocean model (KIST-Ocean) was developed based on a visual attention adversarial network composed of a generator and a discriminator (Guo et al., 2023; Li et al., 2023), designed to simulate the global three-dimensional ocean.

## Repository structure
> <code>KIST-Ocean/</code>: main directory
>> <code>model/</code>
>>> <code>AVAN/</code>: Python scripts for training KIST-Ocean model
>>>
>>>> <code>train_v01.py</code>: Python script for training KIST-OCean
>>>> 
>>>> <code>config.py</code>: Configuration for training and inference
>>>> 
>>>> <code>AVAN_v01.py</code>: Python script for the backbone of the KIST-Ocean model
>>>> 
>>>> <code>utils.py</code>: Python script for containing various utility functions
>>>> 
>>>> <code>inferencer_GT.py</code>: Python script for inferring the future ocean state by prescribing ground truth (observation) as the surface boundary condition (i.e., generating KIST-O_GT)
>>>
>>> <code>output/</code>: Directory where the trained model is saved

>> <code>data/</code>: Statistical datasets required for training and inference

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

