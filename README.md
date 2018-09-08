
Dire is a c++ package that provides three  features:

* Without third-party library,  Deep neural networks built on  pure c++ 
* Double precision  computation  with  GPU acceleration
* As version V0.1, it is too slow to train very deep network.

The following applications are available with dire:
* [bone suppression](https://github.com/qjchen1972/dire/blob/master/bone%20suppression/README.md)
* [Bacilli detection](https://github.com/qjchen1972/dire/blob/master/Bacilli%20detection/README.md)
* [ChexNet with dire ](https://github.com/qjchen1972/dire/blob/master/ChexNet/README.md)



Dependencies
====
* c++11 or later
* cuda7.0 or later (if you need gpu)


Install
===
* GPU:
   * nvcc -m64 -gencode arch=compute_60,code=sm_60 main.cu -o dire_gpu -std=c++11
   * (On windows, run cmd with the Developer Command Prompt for VS 2017)

* CPU:
   * On linux:     g++ main.cpp -o dire -std=c++11
   * On windows:   compile with VS 2017

Dire can generate dynamic libraries or static libraries, and other programs call it.
* [windows dll](https://github.com/qjchen1972/dire/blob/master/dll/README.md)



Getting Started
====

* create data file：train_data  train_label  test_data test_label
    * [one sample with simple tool](https://github.com/qjchen1972/dire/blob/master/data_tool/README.md)
* create net.h and test.h for applications:
    * net.h is networks model 
    * test.h is special test for applications
* compile it
* setup config.txt
* train:
    * dire_gpu 0 config.txt
* test:
    * dire_gpu 2 config.txt
* gradient check:
    * dire_gpu 3 config.txt    

