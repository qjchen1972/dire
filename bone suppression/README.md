
The purpose of this project is to remove the ribs of chest radiographs. After referring to [this paper](https://github.com/qjchen1972/dire/blob/master/bone%20suppression/GZ%20U%20B%20S.pdf), I used gradient images as training data and trained with dire

* network model:
   * The Resnet of 12 layers, refering to [net.h](https://github.com/qjchen1972/dire/blob/master/bone%20suppression/net.h)
   
*  From 100 images of bone chest, randomly cut, and generate 512000 32X32 gradient images. At the same time, the corresponding 8X8 label were generated from the corresponding bone removed chest radiographs:
   * tool xxx/train_data xxx/train_label 32 32 512000
   
*  A bone chest radiograph of 512X512 is divided into N gradient maps of 32X32, input the trained model, generate data, and then solve a Possion equation. Finally, the bony chest film was obtained. 
   * tool src_file  grad_file 512  512 32  32  out.jpg  10

* The original is on the left and the generated graph is on the right: 
![](https://github.com/qjchen1972/dire/blob/master/img/bone_test.png)

