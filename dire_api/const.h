#pragma once
/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: const.h
* 文件标识:
* 摘    要: 一些常量，抽出来的原因是gpu需要用到。
*
*
* 当前版本: 1.0
* 作    者: chen.qian.jiang
* 开始日期: 2017-08-05
* 完成日期:
* 其它说明：
* 修改日期        版本号     修改人          修改内容
* -----------------------------------------------
* 2017/08/05      V1.0       陈黔江          创建
*/

#ifndef _CONST__H
#define _CONST__H

//active
#define RELU     0
#define SIGMOID  1
#define TANH     2
#define SOFTMAX  3
#define LEAKY_RELU 4


#define ACTIVE_PARAM_FILE  "active_"
#define ACTIVE_CHECK_LOG   "log/active_chk_"

//batchnorm
#define BN_PARAM_FILE  "bn_"

//conv
#define CONV_PARAM_FILE  "conv_"
#define CONV_CHECK_LOG   "log/conv_chk_"


//cost function
#define  LSE           0  //最小二乘法
#define  CROSS_ENTROPY 1  //交叉熵
#define  MSE           2  //均方误差
#define  WEIGHT_CROSS_ENTROPY 3  //权值的交叉熵

//grating
#define GRATING_TYPE  0
#define CENTRE_TYPE   1

//limit
#define LIMIT_MAX   0.9999999999
#define LIMIT_MIN   1e-10

//无穷小
#define  INFINIT_NUM   1e-10

// 学习mode
#define  STATIC_STUDY   0   
#define  MOMENTUM_STUDY 1   
#define  DYNAMIC_STUDY  2   //del
#define  ADAM_STUDY     3

// 冲量系数
#define MOMENTUM_RATE   0.5

//用于adam学习的无偏估计的常数
#define MEAN_WEIGTH_RATE 0.9
#define VAR_WEIGTH_RATE  0.999

//正则方式
#define NO_RULE   0 
#define L1_RULE   1
#define L2_RULE   2

#define RULE_RATE  0.0001 
//层的类型
#define GLOBAL_LAYER  "base layer"
#define START_LAYER   "start layer"
#define CONV_LAYER    "conv layer"
#define SAMP_LAYER    "samp layer"
#define BN_LAYER      "bn layer"
#define SCALE_LAYER   "scale layer"
#define GRATING_LAYER "grating layer"
#define ACIVE_LAYER   "active layer"
#define END_LAYER     "end layer"
#define UNSCALED_LAYER "unscaled_layer" 
#define MOVE_LAYER     "move_layer"
#define LIMIT_LAYER    "limit_layer"
#define LINEARMAPPING_LAYER  "linearmapping_layer" 
#define ALINEAR_LAYER  "Alinear"

//初始化参数的方式，原理都一样，都是为了保证var(xk)= var(xk-1)
//以下初始化，尽管采用了通常名称，但加入自己的理解，让bias的方差趋于0，让每层的参数初始化处理都不同，不过目的依旧是保证var(xk)= var(xk-1)
#define XAVIER   0   //采用均匀分布
#define MARS     1   //采用高斯分布,对relu函数作了更精确的处理

//网络运行状态
#define INIT_TRAINING 0  //所有参数初始化进行的训练
#define CONT_TRAINING 1  //继续上一次的参数进行训练
#define TESTING  2
#define PARAM_CHECK    3

//全局参数的备份文件"
#define GLOBAL_PARAM_FILE    "global.txt"
#define PARAM_FILE           "param"

//samp
#define  MAX_POOL   0 
#define  AVG_POOL   1
#define  GLOBAL_AVG_POOL  2   //就是平均池化的特殊
#define  GLOBALAVG_MAX_POOL 3 //输出

#define SAMP_PARAM_FILE  "samp_"
#define SAMP_CHECK_LOG   "log/samp_chk_"

//scale
#define SCALE_PARAM_FILE  "scale_"
#define SCALE_CHECK_LOG   "log/scale_chk_"
//batchnorm
#define GLOBAL_EX         0
#define SINGLE_EX         1

//linearmapping 
#define LINEARMAPPING_PARAM_FILE  "linearmapping_"
#define LINEARMAPPING_CHECK_LOG  "log/linearmapping_chk_"

#define  TRAIN_DATA   "train_data"
#define  TRAIN_LABEL  "train_label"
#define  TEST_DATA    "test_data"
#define  TEST_LABEL   "test_label"
#define  TEST_OUTPUT   "test_output"

#define  PARAM_PRECISION   15 
#define  GTYPE  double


#define RESNET    0     //采用残差网络模式构建
#define DENSENET  1     //采用密集网络模式构建

#define  MAX_LAYER_NUM  1170

// 需要GPUblock*MAX_SUM_MEM >= max fw_buff_size
#define  MAX_SUM_MEM   8*1024*1024

//接入最大的跳转数
#define  MAX_CONN_NUM  128

#define M_PI       3.14159265358979323846

//释放内存的时机
#define FUNC_FREE    0   //函数用完就释放
#define PROC_FREE    1    //一个过程用完就释放
#define NERVER_FREE  2    //永不释放

//保存参数的模式
#define  LOSS_MODE_OFF  0
#define  LOSS_MODE_ON   1
#define  ANSWER_MODE_OFF 0
#define  ANSWER_MODE_ON  1
#define  LAST_MODE_OFF 0
#define  LAST_MODE_ON  1
#endif