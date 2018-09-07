/*
* Copyright (c) 2017
* All rights reserved.
*
* 文件名称: setup.h
* 文件标识:
* 摘    要: 一些设置，以前专注于算法。没有考虑设置方便易用。初5，把这个补一下
*
*
* 当前版本: 1.0
* 作    者: chen.qian.jiang
* 开始日期: 2018-02-20
* 完成日期:
* 其它说明：
* 修改日期        版本号     修改人          修改内容
* -----------------------------------------------
* 2018/02/20      V1.0       陈黔江          创建
*/
#pragma once
#ifndef _SETUP__H
#define _SETUP__H
#include "const.h"

#include<string.h>
#include <iostream>
#include <fstream>

using namespace std;
//#include<stdio.h>

#define  FILENAME_LEN  256


class Setup
{
public:
	Setup(){}
	~Setup() {}


	void set_config_file(char* file)
	{
		strcpy(m_config_file, file);
	}
	
	bool read_param()
	{
		ifstream f;
		f.open(m_config_file, ios::in);
		if (!f.is_open())
		{
			printf("open file %s error!\n", m_config_file);
			return false;
		}

		char tempstr[128];
		f >> tempstr >> m_batch_num;
		f >> tempstr >> m_free_type;
		f >> tempstr >> m_train_dir;
		f >> tempstr >> m_test_dir;

		f >> tempstr >> m_work_param_dir;
		f >> tempstr >> m_net_mode;
		f >> tempstr >> m_initweight_mode;
		f >> tempstr >> m_study_mode;
		f >> tempstr >> m_stu_rate;
		f >> tempstr >> m_regular_mode;
		f >> tempstr >> m_rule_rate;
		f >> tempstr >> m_momentum_rate;

		f >> tempstr >> m_minloss_param_dir;
		f >> tempstr >> m_loss_mode;
		f >> tempstr >> m_last_param_dir;
		f >> tempstr >> m_last_mode;
		f >> tempstr >> m_bestanswer_param_dir;
		f >> tempstr >> m_answer_mode;

		f >> tempstr >> m_test_batch_num;
		f >> tempstr >> m_train_batch_num;
		f >> tempstr >> m_test_mode;
		f >> tempstr >> m_test_log;

		f.close();
		return true;
	}

	//启动程序时有效
	unsigned int m_batch_num = 0;
	int  m_free_type = NERVER_FREE;


	// test
	unsigned int m_test_batch_num = 0;
	unsigned int m_train_batch_num = 0;
	int  m_test_mode = GLOBAL_EX;
	int  m_test_log = 1;

	//目录
	char m_last_param_dir[FILENAME_LEN] = { 0 }; //最新的一次参数
	char m_minloss_param_dir[FILENAME_LEN] = { 0 }; //最小loss的参数
	char m_bestanswer_param_dir[FILENAME_LEN] = { 0 }; //最好结果的参数
	char m_work_param_dir[FILENAME_LEN] = { 0 }; //用于工作的参数
	int  m_last_mode = LAST_MODE_ON;
	int  m_loss_mode = LOSS_MODE_ON;
	int  m_answer_mode = ANSWER_MODE_ON;

	char m_train_dir[FILENAME_LEN] = { 0 };
	char m_test_dir[FILENAME_LEN] = { 0 };

	//全局变量
	int m_study_mode = STATIC_STUDY;   //学习方式
	int m_initweight_mode = MARS; //初始化参数的方式
	int m_regular_mode = NO_RULE; //正则方式
	GTYPE m_momentum_rate = MOMENTUM_RATE;//冲量系数
	int m_net_mode = RESNET; //网络单元构建模式
	GTYPE m_stu_rate = 0.1;  //学习率
	GTYPE m_rule_rate = RULE_RATE; //参数衰减率

private:
	char m_config_file[FILENAME_LEN] = { 0 };
};

Setup g_config;

#endif

