# Hardware accelerated Machine Learning on SoC :rocket:

This project is a __hardware accelerated k-nearest neighbours__ algorithm implemented using the Xilinx Zynq-7000 SoC. The implementation is based on a custom hardware platform developed in Xilinx Vivado, and a co-designed system to run on the platform, developed in Xilinx SDSoC. The system is designed and optimized using HLS and SDS. Software algorithms are implemented using C++11.

__System structure:__
- The accelerator implemented in programmable logic (Artix-7 FPGA) pipelines MAC operations to compute squared distances between a test vector and all reference vectors.
- The processing system (Arm Cortex-A9) performs sorting and searching for the K nearest neighbours, using the vector of squared distances returned from the accelerator. The user can choose the value for K on start-up.
- Data is moved between PL and PS using FastDMA.
- A dataset of 100 test vectors and 1024 query vectors is loaded via a header file. Currently, the system does colour classification of M&Ms using sampled 63-dimensionals feature vectors.

__Documentation:__ 
- A [report](https://github.com/janusboandersen/machine-learning-on-soc/blob/main/report-knn-on-soc.pdf) describing the system and development, and 
- a [slide set](https://github.com/janusboandersen/machine-learning-on-soc/blob/main/slides-knn-on-soc.pdf) introducing the problem, deep-diving into the developed hardware, pros and cons of the co-design development methodolgy and suggestions for future work.
