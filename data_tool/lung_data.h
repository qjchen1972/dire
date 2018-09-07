#pragma once
#ifndef _LUNG_DATA__H
#define _LUNG_DATA__H


#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <string>
#include "tool_api/img_proc.h"
#include "tool_api/data_proc.h"

//#include <io.h>  


using namespace std;

vector<string> files;

#define  LUNG_SRC_ROW  224
#define  LUNG_SRC_COL  224
#define  LUNG_SRC_CHN  2

#define  FILE_LEN  128
#define  SICK_TYPE  15
#define  TWO_TYPE  2

#define LABEL_LEN  14

int sick_train[SICK_TYPE] = { 2406,3112,1000,1214,600,3059,800,700,100,4000,2038,2606,1027,300,2099 };

int sick_all[SICK_TYPE] = {2506,4212,1094,1314,634,3959,895,727,110,9551,2138,2706,1127,307,2199 };

int _train[TWO_TYPE] = {94700 ,1300};


void creat_different_lung_train_data(char* train_file, char* label_file)
{
	ifstream in_file;
	in_file.open("/home/cqj/nih4/dirfiles.txt");
	if (!in_file.is_open())
	{
		printf("open %s error\n", "/home/cqj/nih4/dirfiles.txt");
		return;
	}

	ofstream df;
	df.open(train_file, ios::out | ios::trunc | ios::binary);
	if (!df.is_open())
	{
		printf("open file %s error!\n", train_file);
		return;
	}
	ofstream lf;
	lf.open(label_file, ios::out | ios::trunc | ios::binary);
	if (!lf.is_open())
	{
		printf("open file %s error!\n", label_file);
		return;
	}

	int batch_size = LUNG_SRC_ROW * LUNG_SRC_COL * LUNG_SRC_CHN;
	GTYPE* buf = new GTYPE[batch_size];
	char src_file[FILE_LEN];
	GTYPE label_format[LABEL_LEN] = { 0 };
	vector<string> label;
	vector<int> num;
	char *a = new char[28];
	int b = 0;
	int count = 0;

	while (!in_file.eof())
	{		
		in_file >> a;
		label.push_back(a);
		in_file >> b;
		num.push_back(b);
		count++;
	}
	in_file.close();

	for (int i = 0; i < count; i++)
	{
		cout << "µÚ" << i << "¸ö×Ö·û£º" << label[i] << endl;
		memset(label_format, 0, sizeof(GTYPE)*LABEL_LEN);
		const char *ptr = label[i].c_str();
		while (*ptr != '\0')
		{
			char temp_char[2] = {'0','0'};
			GTYPE temp_num = 0;
			memcpy(temp_char, ptr, sizeof(char)* 2);
			ptr += 2;
			temp_num = atof(temp_char);
			if(temp_num > 0)
			label_format[(int)temp_num-1]++;
		}

		for (int j = 0; j < num[i]; j++)
		{
			sprintf(src_file, "/home/cqj/nih4/%s/train%d.png", label[i].c_str(), j + 1);
			get_data_from_gradimg(buf, src_file, LUNG_SRC_ROW, LUNG_SRC_COL);
			df.write((char *)buf, sizeof(GTYPE)*batch_size);
			lf.write((char *)&label_format, sizeof(GTYPE)*LABEL_LEN);
		}		
	}
	delete[]buf;
	delete[] a;
	df.close();
	lf.close();

}


void get_buf_src(GTYPE* buf,char *path,Mat &image)
{
	image = imread(path,0);
	if (image.rows == 0 || image.cols == 0)
	{
		printf("load src failed: %s\n", path);
		return ;  //¶ÁÈ¡Í¼Æ¬Ê§°Ü
	}

	Mat new_mask(Size(224, 224), CV_8UC1);
	resize(image, new_mask, Size(224, 224));
	image = new_mask;

	for (int i = 0; i < LUNG_SRC_ROW; i++)
	{
		for (int j = 0; j < LUNG_SRC_COL; j++)
		{
			//out[i * col + j] = src.at<uchar>(pos_x + i, pos_y + j);
			buf[i*LUNG_SRC_COL + j] = image.at<uchar>(i, j);
		}
	}
}

void creat_different_lung_train_data_src(char* train_file, char* label_file)
{
	ifstream in_file;
	in_file.open("/home/cqj/nih4/dirfiles.txt");
	if (!in_file.is_open())
	{
		printf("open %s error\n", "/home/cqj/nih4/dirfiles.txt");
		return;
	}

	ofstream df;
	df.open(train_file, ios::out | ios::trunc | ios::binary);
	if (!df.is_open())
	{
		printf("open file %s error!\n", train_file);
		return;
	}
	ofstream lf;
	lf.open(label_file, ios::out | ios::trunc | ios::binary);
	if (!lf.is_open())
	{
		printf("open file %s error!\n", label_file);
		return;
	}

	int batch_size = LUNG_SRC_ROW * LUNG_SRC_COL * LUNG_SRC_CHN;
	GTYPE* buf = new GTYPE[batch_size];
	char src_file[FILE_LEN];
	GTYPE label_format[LABEL_LEN] = { 0 };
	vector<string> label;
	vector<int> num;
	char *a = new char[28];
	int b = 0;
	int count = 0;
	Mat image;
	image = Mat::zeros(LUNG_SRC_ROW, LUNG_SRC_COL, CV_8UC1);

	while (!in_file.eof())
	{
		in_file >> a;
		label.push_back(a);
		in_file >> b;
		num.push_back(b);
		count++;
	}
	in_file.close();

	for (int i = 0; i < count; i++)
	{
		cout << "µÚ" << i << "¸ö×Ö·û£º" << label[i] << endl;
		memset(label_format, 0, sizeof(GTYPE)*LABEL_LEN);
		const char *ptr = label[i].c_str();
		while (*ptr != '\0')
		{
			char temp_char[2] = { '0', '0' };
			GTYPE temp_num = 0;
			memcpy(temp_char, ptr, sizeof(char)* 2);
			ptr += 2;
			temp_num = atof(temp_char);
			if (temp_num > 0)
				label_format[(int)temp_num - 1]++;
		}

		for (int j = 0; j < num[i]; j++)
		{
			sprintf(src_file, "/home/cqj/nih4/%s/train%d.png", label[i].c_str(), j + 1);
			//get_data_from_gradimg(buf, src_file, LUNG_SRC_ROW, LUNG_SRC_COL);
			//image = imread(src_file);
			get_buf_src(buf, src_file, image);
			df.write((char *)buf, sizeof(GTYPE)*batch_size);
			lf.write((char *)&label_format, sizeof(GTYPE)*LABEL_LEN);
		}
	}
	delete[]buf;
	delete[] a;
	df.close();
	lf.close();

}


void lung_create_traing_data(char* train_file, char* label_file,int sick_num , int is_augm = 0)
{
	ofstream df;
	df.open(train_file, ios::out | ios::trunc | ios::binary);
	if (!df.is_open())
	{
		printf("open file %s error!\n", train_file);
		return;
	}
	ofstream lf;
	lf.open(label_file, ios::out | ios::trunc | ios::binary);
	if (!lf.is_open())
	{
		printf("open file %s error!\n", label_file);
		return;
	}	

	char src_file[FILE_LEN];

	int batch_size = LUNG_SRC_ROW * LUNG_SRC_COL * LUNG_SRC_CHN;
	GTYPE* buf = new GTYPE[batch_size];
	GTYPE  ans[SICK_TYPE];	

	int type_num[SICK_TYPE];
	int total = 0;
	memset(type_num, 0, sizeof(int) * SICK_TYPE);	

	time_t now = time(NULL);
	srand(now);

	for (int i = 0; i < SICK_TYPE; i++)
	{
		memset(ans, 0, sizeof(double) * SICK_TYPE);
		ans[i] = 1;
		for (int j = 0; j < sick_num; j++)
		{
			if (type_num[i] >= sick_train[i]) type_num[i] = 0;
			sprintf(src_file, "/home/cqj/nih/%d/train (%d).png", i, type_num[i] + 1);
			get_data_from_gradimg(buf, src_file, LUNG_SRC_ROW, LUNG_SRC_COL);
			df.write((char*)buf, sizeof(double)*batch_size);
			lf.write((char*)ans, sizeof(double) * SICK_TYPE);
			type_num[i]++;
			total++;
		}
		printf("%d finished, total-- %d finish!\n", i, total);
	}
	df.close();
	lf.close();
}

void lung_create_test_data(char* train_file, char* label_file, int sick_num)
{
	ofstream df;
	df.open(train_file, ios::out | ios::trunc | ios::binary);
	if (!df.is_open())
	{
		printf("open file %s error!\n", train_file);
		return;
	}
	ofstream lf;
	lf.open(label_file, ios::out | ios::trunc | ios::binary);
	if (!lf.is_open())
	{
		printf("open file %s error!\n", label_file);
		return;
	}

	char src_file[FILE_LEN];

	int batch_size = LUNG_SRC_ROW * LUNG_SRC_COL * LUNG_SRC_CHN;
	GTYPE* buf = new GTYPE[batch_size];
	GTYPE  ans[SICK_TYPE];

	int type_num[SICK_TYPE];
	int total = 0;
	memset(type_num, 0, sizeof(int) * SICK_TYPE);
	for (int i = 0; i < SICK_TYPE; i++)
		type_num[i] = sick_train[i];

	for (int i = 0; i < SICK_TYPE; i++)
	{
		memset(ans, 0, sizeof(GTYPE) * SICK_TYPE);
		ans[i] = 1;
		for (int j = 0; j < sick_num; j++)
		{
			if (type_num[i] >= sick_all[i]) break;
			sprintf(src_file, "/home/cqj/nih/%d/train (%d).png", i, type_num[i] + 1);
			//sprintf(src_file, "srcdata/%d/train%d.png", i, type_num[i] + 1);
			get_data_from_gradimg(buf, src_file, LUNG_SRC_ROW, LUNG_SRC_COL);
			df.write((char*)buf, sizeof(GTYPE)*batch_size);
			lf.write((char*)ans, sizeof(GTYPE) * SICK_TYPE);
			type_num[i]++;
			total++;
		}
		printf("%d finished, total-- %d finish!\n", i, total);
	}
	df.close();
	lf.close();
}


void lung_create_one_test(char *srcfile, char *datafile, char* labelfile, int type)
{
	ofstream df, lf;
	df.open(datafile, ios::out | ios::trunc | ios::binary);
	if (!df.is_open())
	{
		printf("open file %s error!\n", datafile);
		return;
	}

	lf.open(labelfile, ios::out | ios::trunc | ios::binary);
	if (!lf.is_open())
	{
		printf("open file %s error!\n", labelfile);
		return;
	}

	GTYPE  ans[1];
	//memset(ans, 0, sizeof(GTYPE) *1);
	ans[0] = type;
	lf.write((char*)ans, sizeof(GTYPE) * 1);
	lf.close();

	int batch_size = LUNG_SRC_ROW * LUNG_SRC_COL * LUNG_SRC_CHN;
	GTYPE* buf = new GTYPE[batch_size];
	get_data_from_gradimg(buf, srcfile, LUNG_SRC_ROW, LUNG_SRC_COL);
	//get_data_from_img(buf, srcfile, SRC_ROW, SRC_COL);
	df.write((char*)buf, sizeof(GTYPE)*batch_size);
	df.close();
	delete[] buf;
}



void create_traing_data(char* train_file, char* label_file )
{
	ofstream df;
	df.open(train_file, ios::out | ios::trunc | ios::binary);
	if (!df.is_open())
	{
		printf("open file %s error!\n", train_file);
		return;
	}
	ofstream lf;
	lf.open(label_file, ios::out | ios::trunc | ios::binary);
	if (!lf.is_open())
	{
		printf("open file %s error!\n", label_file);
		return;
	}	

	char src_file[FILE_LEN];

	int batch_size = LUNG_SRC_ROW * LUNG_SRC_COL * LUNG_SRC_CHN;
	GTYPE* buf = new GTYPE[batch_size];
	GTYPE  ans[1];	

	int type_num[TWO_TYPE];
	int total = 0;
	memset(type_num, 0, sizeof(int) * TWO_TYPE);	

	time_t now = time(NULL);
	srand(now);
	int sick_num = 0;
	for (int i = 0; i < TWO_TYPE; i++)
	{	
		sick_num = _train[i];	
		//memset(ans, 0, sizeof(double) * 1);
		ans[0] = i;
		for (int j = 0; j < sick_num; j++)
		{
			sprintf(src_file, "/home/cqj/nih3/%d/train (%d).png", i, type_num[i] + 1);
			get_data_from_gradimg(buf, src_file, LUNG_SRC_ROW, LUNG_SRC_COL);
			df.write((char*)buf, sizeof(GTYPE)*batch_size);
			lf.write((char*)ans, sizeof(GTYPE) * 1);
			type_num[i]++;
			total++;
		}
		printf("%d finished, total-- %d finish!\n", i, total);
	}
	df.close();
	lf.close();
}


#endif
