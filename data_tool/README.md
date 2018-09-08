
* On linux,  build tool: 
  * g++ -o ../tool  main.cpp `pkg-config --cflags --libs opencv`
* create  "train data and train label" of the bone suppression :
  * First, the dicom images is in the directory "bone/traindata/"
  * Then Produce 512000 images of 32*32 :  tool xxx/train_data xxx/train_label 32  32  512000
  
