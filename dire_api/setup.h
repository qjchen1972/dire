/*
* Copyright (c) 2017
* All rights reserved.
*
* �ļ�����: setup.h
* �ļ���ʶ:
* ժ    Ҫ: һЩ���ã���ǰרע���㷨��û�п������÷������á���5���������һ��
*
*
* ��ǰ�汾: 1.0
* ��    ��: chen.qian.jiang
* ��ʼ����: 2018-02-20
* �������:
* ����˵����
* �޸�����        �汾��     �޸���          �޸�����
* -----------------------------------------------
* 2018/02/20      V1.0       ��ǭ��          ����
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

	//��������ʱ��Ч
	unsigned int m_batch_num = 0;
	int  m_free_type = NERVER_FREE;


	// test
	unsigned int m_test_batch_num = 0;
	unsigned int m_train_batch_num = 0;
	int  m_test_mode = GLOBAL_EX;
	int  m_test_log = 1;

	//Ŀ¼
	char m_last_param_dir[FILENAME_LEN] = { 0 }; //���µ�һ�β���
	char m_minloss_param_dir[FILENAME_LEN] = { 0 }; //��Сloss�Ĳ���
	char m_bestanswer_param_dir[FILENAME_LEN] = { 0 }; //��ý���Ĳ���
	char m_work_param_dir[FILENAME_LEN] = { 0 }; //���ڹ����Ĳ���
	int  m_last_mode = LAST_MODE_ON;
	int  m_loss_mode = LOSS_MODE_ON;
	int  m_answer_mode = ANSWER_MODE_ON;

	char m_train_dir[FILENAME_LEN] = { 0 };
	char m_test_dir[FILENAME_LEN] = { 0 };

	//ȫ�ֱ���
	int m_study_mode = STATIC_STUDY;   //ѧϰ��ʽ
	int m_initweight_mode = MARS; //��ʼ�������ķ�ʽ
	int m_regular_mode = NO_RULE; //����ʽ
	GTYPE m_momentum_rate = MOMENTUM_RATE;//����ϵ��
	int m_net_mode = RESNET; //���絥Ԫ����ģʽ
	GTYPE m_stu_rate = 0.1;  //ѧϰ��
	GTYPE m_rule_rate = RULE_RATE; //����˥����

private:
	char m_config_file[FILENAME_LEN] = { 0 };
};

Setup g_config;

#endif

