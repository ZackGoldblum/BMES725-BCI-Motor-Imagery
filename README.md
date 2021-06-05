# BMES 725 BCI Motor Imagery
This is the code for the BMES 725 term project "Real-time Classification of Motor Imagery Data via an EEG-based Brain Computer Interface" by Zack Goldblum, Dan Thompson, and Adam Wojnar. 

In this project we evaluate the validity of three convolutional neural networks ([EEGNet](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c), [Shallow ConvNet](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), and [Deep ConvNet](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)) for the real-time classification of motor imagery tasks. Adapated verions of these models are from the [Army Research Laboratory (ARL) EEGModels project](https://github.com/vlawhern/arl-eegmodels).

The [OpenBCI Ultracortex Mark IV](https://docs.openbci.com/docs/04AddOns/01-Headwear/MarkIV) and [Cyton Biosensing Board](https://docs.openbci.com/docs/02Cyton/CytonLanding) were used as the EEG headset and data acquisition board in both the creation of our motor imagery dataset and in our real-time classification program. EEG data was acquired from the 8 electrode channels with the following 10-20 System locations at a 250 Hz sampling rate:

![image](https://user-images.githubusercontent.com/18644336/120905586-041a8780-c621-11eb-91ee-abac811d6749.png)
