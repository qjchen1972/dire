#pragma once
#ifndef _BONE__H
#define _BONE__H

#include<time.h>
#include <iostream>
#include <fstream>
#include "tool_api/img_proc.h"
#include "tool_api/possion.h"

using namespace cv;
using namespace std;

#define  BONE_DATA_ROW  32
#define  BONE_DATA_COL  32
#define  BONE_LABEL_ROW  24
#define  BONE_LABEL_COL  24

#define  FILE_LEN  128
#define  BONE_IMG_NUM   50




static void  get_all_file(Mat *Ix, Mat *Iy, Mat *X, Mat *Y, ofstream &df, ofstream &lf,int img_num)
{
	int row = Ix[0].rows;
	int col = Ix[0].cols;
	GTYPE  *value = new GTYPE[BONE_DATA_ROW*BONE_DATA_COL];

	int count = 0;

	while (count < img_num)
	{
		int pos = rand() % BONE_IMG_NUM;
		int select = rand() % 10;
		int r_x, r_y;
		if (select < 4)
		{
			r_x = rand() % (row - BONE_DATA_ROW);
			r_y = rand() % (col - BONE_DATA_COL);
		}
		else
		{
			r_x = 50 + rand() % (row - BONE_DATA_ROW - 80);
			r_y = 50 + rand() % (col - BONE_DATA_COL - 80);
		}

		for (int i = 0; i < BONE_DATA_ROW; i++)
			for (int j = 0; j < BONE_DATA_COL; j++)
				value[i*BONE_DATA_COL + j] = Ix[pos].at<float>(r_x + i, r_y + j);
		df.write((char*)value, sizeof(GTYPE)*BONE_DATA_ROW*BONE_DATA_COL);

		for (int i = 0; i < BONE_DATA_ROW; i++)
			for (int j = 0; j < BONE_DATA_COL; j++)
				value[i*BONE_DATA_COL + j] = Iy[pos].at<float>(r_x + i, r_y + j);
		df.write((char*)value, sizeof(GTYPE)*BONE_DATA_ROW*BONE_DATA_COL);

		int l_x = r_x + BONE_LABEL_ROW / 2;
		int l_y = r_y + BONE_LABEL_COL / 2;

		for (int i = 0; i < BONE_DATA_ROW - BONE_LABEL_ROW; i++)
			for (int j = 0; j < BONE_DATA_COL - BONE_LABEL_COL; j++)
				value[i*(BONE_DATA_COL - BONE_LABEL_COL) + j] = X[pos].at<float>(l_x + i, l_y + j);
		lf.write((char*)value, sizeof(GTYPE)*(BONE_DATA_ROW - BONE_LABEL_ROW)*(BONE_DATA_COL - BONE_LABEL_COL));

		for (int i = 0; i < BONE_DATA_ROW - BONE_LABEL_ROW; i++)
			for (int j = 0; j < BONE_DATA_COL - BONE_LABEL_COL; j++)
				value[i*(BONE_DATA_COL - BONE_LABEL_COL) + j] = Y[pos].at<float>(l_x + i, l_y + j);
		lf.write((char*)value, sizeof(GTYPE)*(BONE_DATA_ROW - BONE_LABEL_ROW)*(BONE_DATA_COL - BONE_LABEL_COL));
		count++;
	}
}

void bone_create_train_data(char* train_data, char*  train_label,int total_row, int total_col,int num)
{
	ofstream df;
	df.open(train_data, ios::out | ios::trunc | ios::binary);
	if (!df.is_open())
	{
		printf("open file %s error!\n", train_data);
		return;
	}
	ofstream lf;
	lf.open(train_label, ios::out | ios::trunc | ios::binary);
	if (!lf.is_open())
	{
		printf("open file %s error!\n", train_label);
		return;
	}

	time_t now = time(NULL);
	srand(now);

	char src_file[FILE_LEN];
	char label_file[FILE_LEN];

	Mat sIx[BONE_IMG_NUM];
	Mat sIy[BONE_IMG_NUM];
	Mat sX[BONE_IMG_NUM];
	Mat sY[BONE_IMG_NUM];

	Mat src, nosrc;
	Size s = Size(total_row, total_col);
	Possion pos;
	pos.set_outchan(1);

	for (int i = 0; i < BONE_IMG_NUM; i++)
	{
		sprintf(label_file, "bone/traindata/train (%d).dcm", 2 * i + 1);
		sprintf(src_file, "bone/traindata/train (%d).dcm", 2 * i + 2);
		read_dicom(src_file, src);

		Mat new_src = Mat(s, CV_8UC1);
		resize(src, new_src, s);
		src = new_src;
		Mat  X;
		pos.computeGradientX(src, X);
		//得到垂直图像梯度
		Mat  Y;
		pos.computeGradientY(src, Y);

		X.copyTo(sIx[i]);
		Y.copyTo(sIy[i]);

		read_dicom(label_file, nosrc);
		Mat new_src1 = Mat(s, CV_8UC1);
		resize(nosrc, new_src1, s);
		nosrc = new_src1;
		Mat nX, nY;
		pos.computeGradientX(nosrc, nX);
		pos.computeGradientY(nosrc, nY);
		nX.copyTo(sX[i]);
		nY.copyTo(sY[i]);
	}
	get_all_file(sIx, sIy, sX, sY, df, lf,num);
	df.close();
	lf.close();
}

void bone_create_one_test(char *test_file, char*  src_label, int total_row, int total_col, int row,int col,char *test_out, char *test_label)
{
	ofstream tf, tl;
	tf.open(test_out, ios::out | ios::trunc | ios::binary);
	if (!tf.is_open())
	{
		printf("open file %s error!\n", test_out);
		return;
	}

	tl.open(test_label, ios::out | ios::trunc | ios::binary);
	if (!tl.is_open())
	{
		printf("open file %s error!\n", test_label);
		return;
	}

	Mat src, nosrc;
	// read dcm
	read_dicom(test_file, src);
	read_dicom(src_label, nosrc);

	Size s = Size(total_row, total_col);
	Mat new_src = Mat(s, CV_8UC1);
	resize(src, new_src, s);
	src = new_src;

	Mat new_nosrc = Mat(s, CV_8UC1);
	resize(nosrc, new_nosrc, s);
	nosrc = new_nosrc;

	Possion pos;
	pos.set_outchan(1);

	Mat  X, nX;
	pos.computeGradientX(src, X);
	pos.computeGradientX(nosrc, nX);

	//得到垂直图像梯度
	Mat  Y, nY;
	pos.computeGradientY(src, Y);
	pos.computeGradientY(nosrc, nY);

	//Mat I;
	//I = X+Y;

	GTYPE  value;
	// 求解一个方程 512 - x = m * ( x -24 ）
	// 488/ (m+1) = x-24

	GTYPE *block = new GTYPE[row*col];
	GTYPE *label = new GTYPE[(row - BONE_LABEL_ROW)*(col - BONE_LABEL_COL)];

	for (int i = 0; i <= total_row - row; i += row - BONE_LABEL_ROW)
	{
		for (int j = 0; j <= total_col - col; j += col - BONE_LABEL_COL)
		{
			for (int m = 0; m < row; m++)
				for (int n = 0; n < col; n++)
					block[m*col + n] = X.at<float>(i + m, j + n);
			tf.write((char*)block, sizeof(GTYPE)*row*col);

			for (int m = 0; m < row; m++)
				for (int n = 0; n < col; n++)
					block[m*col + n] = Y.at<float>(i + m, j + n);
			tf.write((char*)block, sizeof(GTYPE)*row*col);
			

			for (int m = 0; m < row - BONE_LABEL_ROW; m++)
				for (int n = 0; n < col - BONE_LABEL_COL; n++)
					label[m*(col - BONE_LABEL_COL) + n] = nX.at<float>(i + BONE_LABEL_ROW / 2 + m, j + BONE_LABEL_COL / 2 + n);
			tl.write((char*)label, sizeof(GTYPE)*(row - BONE_LABEL_ROW)*(col - BONE_LABEL_COL));

			for (int m = 0; m < row - BONE_LABEL_ROW; m++)
				for (int n = 0; n < col - BONE_LABEL_COL; n++)
					label[m*(col - BONE_LABEL_COL) + n] = nY.at<float>(i + BONE_LABEL_ROW / 2 + m, j + BONE_LABEL_COL / 2 + n);
			tl.write((char*)label, sizeof(GTYPE)*(row - BONE_LABEL_ROW)*(col - BONE_LABEL_COL));
		}
	}
	tf.close();
	tl.close();
	delete[] block;
	delete[] label;
}


void bone_create_one_test_img(char *src_file, char *grad_file, int total_row, int total_col, int row,int col,char* jpg, int light)
{
	ifstream  fdata;
	fdata.open(grad_file, ios::in | ios::binary);
	if (!fdata.is_open())
	{
		printf("open %s error\n", grad_file);
		return;
	}

	GTYPE *data = new GTYPE[2 * (row - BONE_LABEL_ROW)*(col - BONE_LABEL_COL)];

	Mat src;
	// read dcm
	read_dicom(src_file, src);

	Possion pos;
	pos.set_outchan(1);
	
	Size s = Size(total_row,total_col);
	Mat new_src = Mat(s, CV_8UC1);

	resize(src, new_src, s);
	src = new_src;
	
	Mat  X;
	pos.computeGradientX(src, X);
	
	Mat  Y;
	pos.computeGradientY(src, Y);

	
	int count_x = 0;
	int count_y = 0;
	Mat gx, gy;

	gx = Mat(src.size(), CV_32FC1);
	gy = Mat(src.size(), CV_32FC1);

	X.copyTo(gx);
	Y.copyTo(gy);

	while (1)
	{
		if (count_y > src.cols - col)
		{
			count_y = 0;
			count_x += row - BONE_LABEL_ROW;
		}
		if (count_x > src.rows - row) break;
		fdata.read((char*)data, 2 * (row - BONE_LABEL_ROW)*(col - BONE_LABEL_COL) * sizeof(GTYPE));
		if (fdata.eof())		break;
		for (int m = 0; m < row - BONE_LABEL_ROW; m++)
			for (int n = 0; n < col - BONE_LABEL_COL; n++)
			{
				gx.at<float>(count_x + BONE_LABEL_ROW / 2 + m, count_y + BONE_LABEL_COL / 2 + n) = data[m*(col - BONE_LABEL_COL) + n];
				gy.at<float>(count_x + BONE_LABEL_ROW / 2 + m, count_y + BONE_LABEL_COL / 2 + n) = 
					data[(row - BONE_LABEL_ROW)*(col - BONE_LABEL_COL) + m*(col - BONE_LABEL_COL) + n];
			}
		count_y += col - BONE_LABEL_COL;
	}
	fdata.close();

	float lt = light / 10.0;
	gx = gx.mul(lt);
	gy = gy.mul(lt);
	Mat dst;
	pos.evaluate(src, gx, gy, dst);
	imwrite(jpg, dst);
}

#endif

