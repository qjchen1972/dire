#pragma once
/*
* Copyright (c) 2017
* All rights reserved.
*
* �ļ�����: const.h
* �ļ���ʶ:
* ժ    Ҫ: һЩ�������������ԭ����gpu��Ҫ�õ���
*
*
* ��ǰ�汾: 1.0
* ��    ��: chen.qian.jiang
* ��ʼ����: 2017-08-05
* �������:
* ����˵����
* �޸�����        �汾��     �޸���          �޸�����
* -----------------------------------------------
* 2017/08/05      V1.0       ��ǭ��          ����
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
#define  LSE           0  //��С���˷�
#define  CROSS_ENTROPY 1  //������
#define  MSE           2  //�������
#define  WEIGHT_CROSS_ENTROPY 3  //Ȩֵ�Ľ�����

//grating
#define GRATING_TYPE  0
#define CENTRE_TYPE   1

//limit
#define LIMIT_MAX   0.9999999999
#define LIMIT_MIN   1e-10

//����С
#define  INFINIT_NUM   1e-10

// ѧϰmode
#define  STATIC_STUDY   0   
#define  MOMENTUM_STUDY 1   
#define  DYNAMIC_STUDY  2   //del
#define  ADAM_STUDY     3

// ����ϵ��
#define MOMENTUM_RATE   0.5

//����adamѧϰ����ƫ���Ƶĳ���
#define MEAN_WEIGTH_RATE 0.9
#define VAR_WEIGTH_RATE  0.999

//����ʽ
#define NO_RULE   0 
#define L1_RULE   1
#define L2_RULE   2

#define RULE_RATE  0.0001 
//�������
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

//��ʼ�������ķ�ʽ��ԭ��һ��������Ϊ�˱�֤var(xk)= var(xk-1)
//���³�ʼ�������ܲ�����ͨ�����ƣ��������Լ�����⣬��bias�ķ�������0����ÿ��Ĳ�����ʼ��������ͬ������Ŀ�������Ǳ�֤var(xk)= var(xk-1)
#define XAVIER   0   //���þ��ȷֲ�
#define MARS     1   //���ø�˹�ֲ�,��relu�������˸���ȷ�Ĵ���

//��������״̬
#define INIT_TRAINING 0  //���в�����ʼ�����е�ѵ��
#define CONT_TRAINING 1  //������һ�εĲ�������ѵ��
#define TESTING  2
#define PARAM_CHECK    3

//ȫ�ֲ����ı����ļ�"
#define GLOBAL_PARAM_FILE    "global.txt"
#define PARAM_FILE           "param"

//samp
#define  MAX_POOL   0 
#define  AVG_POOL   1
#define  GLOBAL_AVG_POOL  2   //����ƽ���ػ�������
#define  GLOBALAVG_MAX_POOL 3 //���

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


#define RESNET    0     //���òв�����ģʽ����
#define DENSENET  1     //�����ܼ�����ģʽ����

#define  MAX_LAYER_NUM  1170

// ��ҪGPUblock*MAX_SUM_MEM >= max fw_buff_size
#define  MAX_SUM_MEM   8*1024*1024

//����������ת��
#define  MAX_CONN_NUM  128

#define M_PI       3.14159265358979323846

//�ͷ��ڴ��ʱ��
#define FUNC_FREE    0   //����������ͷ�
#define PROC_FREE    1    //һ������������ͷ�
#define NERVER_FREE  2    //�����ͷ�

//���������ģʽ
#define  LOSS_MODE_OFF  0
#define  LOSS_MODE_ON   1
#define  ANSWER_MODE_OFF 0
#define  ANSWER_MODE_ON  1
#define  LAST_MODE_OFF 0
#define  LAST_MODE_ON  1
#endif