
The aim of this project is to identify bacilli and mark their location in images produced by an electron microscope. 

* This is a original  picture of the microscope after processing the picture.

![](https://github.com/qjchen1972/dire/blob/master/img/gan.png)

* We divide the original picture into 64x64 small pictures in sequence. According to the number of bacteria in the picture, it can be divided into 5 categories. The 4 category indicates that the number of bacteria is greater than or equal to 4.

![](https://github.com/qjchen1972/dire/blob/master/img/gan64.png)

* net model:

  The model is modified on a 121-layer densenet with 64*64 gray-scale image input and cross-entropy with weights as loss function. Details refer to [net.h](https://github.com/qjchen1972/dire/blob/master/Bacilli%20detection/net.h)
  
  
*To make a comparison, I asked my colleagues to use tensoflow to train, and I used DIre. As a result, the accuracy of using double precision dire on the test set is 88%, while tensoflow is only 75%. 
