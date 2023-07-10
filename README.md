# CNN-Exercise-Recognition

This research investigates the difference in inferencing speeds given various models on several types of Kria Xiling boards. The main code is written by the head researcher in C++ and executed on Xilinx Kria boards, while the code written in this repository is in python and executed on a host PC in order to test the capabilities of the NVIDIA Geforce 3070 GPU (in comparison to the Kria Boards).

- The model architectures was designed by the lead researcher
- The dataset used for training, testing, and validation all come from the mm-fit dataset (see https://github.com/KDMStromback/mm-fit)
- CMU's Openpose Algorithm (https://github.com/CMU-Perceptual-Computing-Lab/openpose) was used to extract the skeleton features used for training and inferencing
- the original training dataset is skeleton only, not RGB imaging, so Openpose isn't used in this repository
