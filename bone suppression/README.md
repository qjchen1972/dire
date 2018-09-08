
The purpose of this project is to remove the ribs of chest radiographs. After referring to [this paper](https://github.com/qjchen1972/dire/blob/master/bone%20suppression/GZ%20U%20B%20S.pdf), I used gradient images as training data and trained with dire

* network model:
  * The Resnet of 12 layers, refering to [net.h](https://github.com/qjchen1972/dire/blob/master/bone%20suppression/net.h)
*  From 100 images of bone chest, randomly cut, and generate 512000 32*32 gradient images. At the same time, the corresponding 8*8 label were generated from the corresponding bone removed chest radiographs.
  * tool xxx/train_data xxx/train_label 32 32 512000

使用训练后的模型，进行测试得到的结果图：

![这是测试结果](https://github.com/qjchen1972/dire/blob/master/img/bone_test.png)

