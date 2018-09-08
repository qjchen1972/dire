
Dire is a c++ package that provides three  features:

* Without third-party library,  Deep neural networks built on  pure c++ 
* Double precision  computation  with  GPU acceleration
* As version V0.1, it is too slow to train too deep network.

The following applications are available with dire:
* [bone suppression](https://github.com/qjchen1972/dire/blob/master/bone%20suppression/README.md)
* [Bacilli detection](https://github.com/qjchen1972/dire/blob/master/Bacilli%20detection/README.md)
* [ChexNet with dire ](https://github.com/qjchen1972/dire/blob/master/ChexNet/README.md)

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

create data file
---

dire是什么
====

   dire是一个深度学习框架，类似caffe的那种前后向的层的构架。名字来源于dire straits.
   
   2016年，我开始学习突然兴起的深度学习，过程中，为了消化论文，决定自己做了一个深度学习框架。在2017年7月，在这框架上又加入了gpu.在2017年10月变成一个可以做训练的一个比较粗陋的产品。 从20171年11月到2017年2月，我一直坚持用它训练了3个产品。胸片去骨，显微镜下杆菌的辨析，以及ChexNet.
   
   2018年2月后，我就不再使用它了。因为它真的太慢了。更好的原因是，pytorch+onnx +caffe2确实蛮出色，我没有动力去坚持了。

dire的特点
===
  
  1. 它是一个双精度的学习框架。gpu上为了支持双精度，我是一直采取全局内存的方式。没有采用目前只支持单精度的纹理内存。这保持了整个计算过程都是双精度。但全局内存确实太慢了。。。
  2. 它是前向和后向层层递推架构。 中间过程的求导都是需要事先做好推导。
  3. 它太慢了。由于一开始就是为了算法实现而写，优化也是在代码层上。 没有加入第三方库，如openblas,mkl等。
  4. 它是很纯的c++写的，不需要任何修改，可直接在windows, linux等系统下使用
  5. 它对我很自由，我能使用它，随意的测试最新的论文算法。
   
 
dire的几个应用
=====

bone suppression
-------
把胸片的肋骨去掉。dire的双精度让函数回归变得更精细。当时采用的12层类似残差网络。效果如下：
![](https://github.com/qjchen1972/dire/blob/master/img/bone_test.png)
