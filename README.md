Master Project Submission
-------------------------

The uploaded code in this repository is a Tensorflow implementation of the 3D CSPN module from the published work of Cheng et al. "Learning Depth with Convolutional Spatial Propagation Networks" (https://ieeexplore.ieee.org/document/8869936).

- I have trained the code on the entire Scene Flow Dataset (https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). The Scene Flow Dataset consists of three sub-datasets: Driving, FlyingThings3D and Monkaa. 
- I have first downloaded the cleanpass image pairs and the disparity images for these and uploaded them on Kaggle.
- I have then converted the depth map to images and uploaded them. 
- All images from a specific sub-dataset were kept in one folder for easy data-loading.
- I have trained the code on these three sub-datasets separately in parallel on different Kaggle kernels.

Order of execution for successful implementation is as follows:
- utils/IO.py
- models/model_utils.py
- CSPN.py
- models/model.py
- utils/data_loader.py
- train.py
- predict.py

Kaggle already provides all modules required for implementing this code. An important point to be noted: this code is writtedn in Tensorflow v1.8.