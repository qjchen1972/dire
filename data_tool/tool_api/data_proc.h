#pragma once
#ifndef _DATA_PROC__H
#define _DATA_PROC__H

#include <iostream>
#include <fstream>
#include <vector>
#include<time.h>
#include "const.h"
using namespace std;

/* 输入数据多样性
x 是增强后的水平坐标
y 是增强后的垂直坐标
sx 是增强前的水平坐标
sy 是增强前的垂直坐标
*/

/*
水平翻转

sx = (col-1) - x
sy = y
*/
void hor_rot(GTYPE *in, int row, int col, int chn)
{
	int node_size = row * col;
	for (unsigned int j = 0; j < node_size; j++)
	{
		int node_x = j / col;
		int node_y = j % col;
		if (node_y < col / 2)
		{
			for (int k = 0; k < chn; k++)
			{
				GTYPE temp = in[k* node_size + j];
				in[k * node_size + j] = in[k * node_size + node_x * col + col - node_y - 1];
				in[k * node_size + node_x * col + col - node_y - 1] = temp;
			}
		}
	}
}

/*
垂直翻转
sx = x
sy = (row-1) - y
*/
void ver_rot(GTYPE *in, int row, int col, int chn)
{
	int node_size = row * col;
	for (unsigned int j = 0; j < node_size; j++)
	{
		int node_x = j / col;
		int node_y = j % col;
		if (node_x < row / 2)
		{
			for (int k = 0; k < chn; k++)
			{
				GTYPE temp = in[k* node_size + j];
				in[k * node_size + j] = in[k * node_size + (row - node_x - 1) * col + node_y];
				in[k * node_size + (row - node_x - 1) * col + node_y] = temp;
			}
		}
	}
}

/*
平移
sx = x + dx
sy = y + dy
*/
void move(GTYPE *in, int row, int col, int chn,int move_x, int move_y)
{
	int node_size = row * col;
	int batch_size = node_size*chn;
	GTYPE *tempf = new GTYPE[batch_size];
	memcpy(tempf, in, batch_size * sizeof(GTYPE));

	for (unsigned int j = 0; j < node_size; j++)
	{
		int node_x = j / col;
		int node_y = j % col;
		for (int k = 0; k < chn; k++)
		{
			if (node_x + move_x < 0 || node_y + move_y < 0 || node_x + move_x >= row || node_y + move_y >= col)
			{
				in[k * node_size + j] = 0;
			}
			else
			{
				in[k * node_size + j] = tempf[k * node_size + (node_x + move_x) * col + node_y + move_y];
			}
		}
	}
	delete[] tempf;
}


/*
旋转
sx = x*cosa + y*sina //水平
sy = y*cosa - x*sina //垂直
*/
void rotate(GTYPE *in, int row, int col, int chn, int degree)
{
	GTYPE angle = degree * PI / 180.0;
	int node_size = row * col;
	int batch_size = node_size*chn;

	GTYPE *tempf = new GTYPE[batch_size];
	memcpy(tempf, in, batch_size * sizeof(GTYPE));

	int center_x = col % 2 == 1 ? col / 2 + 1 : col / 2;
	int center_y = row % 2 == 1 ? row / 2 + 1 : row / 2;

	for (unsigned int j = 0; j < node_size; j++)
	{
		int node_x = j % col; //水平
		int node_y = j / col; //垂直			
		for (int k = 0; k < chn; k++)
		{
			int src_x = (node_x - center_x)  * cos(angle) + (node_y - center_y)  * sin(angle) + center_x;//水平
			int src_y = (node_y - center_y) * cos(angle) - (node_x - center_x)* sin(angle) + center_y;//垂直

			if (src_x < 0 || src_y < 0 || src_y >= row || src_x >= col)
			{
				in[k * node_size + j] = 0;
			}
			else
			{
				in[k * node_size + j] = tempf[k * node_size + src_y * col + src_x];
			}
		}
	}
	delete[] tempf;
}

/*
扭曲整块的某一象限
pos ：象限
degree : 角度
*/
void twist(GTYPE *in, int row, int col, int chn, int pos, int degree)
{
	int node_size = row * col;
	int bk_row = row / 2;
	int bk_col = col / 2;
	GTYPE *block = new GTYPE[bk_row * bk_col*chn];

	int start_row, start_col, end_row, end_col;

	if (pos == 1)
	{
		start_row = 0;
		start_col = 0;
		end_row = row / 2;
		end_col = col / 2;
	}
	else if (pos == 2)
	{
		start_row = 0;
		start_col = col % 2 == 1 ? col / 2 + 1 : col / 2;
		end_row = row / 2;
		end_col = col;
	}
	else if (pos == 3)
	{
		start_row = row % 2 == 1 ? row / 2 + 1 : row / 2;
		start_col = col % 2 == 1 ? col / 2 + 1 : col / 2;
		end_row = row;
		end_col = col;
	}
	else //pos =4
	{
		start_row = row % 2 == 1 ? row / 2 + 1 : row / 2;
		start_col = 0;
		end_row = row;
		end_col = col / 2;
	}

	for (int i = start_row; i< end_row; i++)
		for (int j = start_col; j < end_col; j++)
			for (int k = 0; k < chn; k++)
				block[k*bk_row * bk_col + (i - start_row)*bk_col + (j - start_col)] = in[k * node_size + i*col + j];
	rotate(block, bk_row, bk_col, chn, degree);

	for (int i = start_row; i< end_row; i++)
		for (int j = start_col; j < end_col; j++)
			for (int k = 0; k < chn; k++)
				in[k * node_size + i*col + j] = block[k*bk_row * bk_col + (i - start_row)*bk_col + (j - start_col)];
	delete[] block;
}

/*
扭曲整块的某一小块的某一象限
pos ：象限
degree : 角度
*/
void twist(GTYPE *in, int row, int col, int chn, int start_x, int start_y, int s_row, int s_col, int pos, int degree)
{
	int node_size = row * col;

	int bk_row = s_row / 2;
	int bk_col = s_col / 2;
	GTYPE *block = new GTYPE[bk_row * bk_col*chn];

	int start_row, start_col, end_row, end_col;

	if (pos == 1)
	{
		start_row = start_x;
		start_col = start_y;
		end_row = start_x + s_row / 2;
		end_col = start_y + s_col / 2;
	}
	else if (pos == 2)
	{
		start_row = start_x;
		start_col = start_y + (s_col % 2 == 1 ? s_col / 2 + 1 : s_col / 2);
		end_row = start_x + s_row / 2;
		end_col = start_y + s_col;
	}
	else if (pos == 3)
	{
		start_row = start_x + (s_row % 2 == 1 ? s_row / 2 + 1 : s_row / 2);
		start_col = start_y + (s_col % 2 == 1 ? s_col / 2 + 1 : s_col / 2);
		end_row = start_x + s_row;
		end_col = start_y + s_col;
	}
	else //pos =4
	{
		start_row = start_x + (s_row % 2 == 1 ? s_row / 2 + 1 : s_row / 2);
		start_col = start_y;
		end_row = start_x + s_row;
		end_col = start_y + s_col / 2;
	}

	for (int i = start_row; i< end_row; i++)
		for (int j = start_col; j < end_col; j++)
			for (int k = 0; k < chn; k++)
				block[k*bk_row * bk_col + (i - start_row)*bk_col + (j - start_col)] = in[k * node_size + i*col + j];
	rotate(block, bk_row, bk_col, chn, degree);

	for (int i = start_row; i< end_row; i++)
		for (int j = start_col; j < end_col; j++)
			for (int k = 0; k < chn; k++)
				in[k * node_size + i*col + j] = block[k*bk_row * bk_col + (i - start_row)*bk_col + (j - start_col)];
	delete[] block;
}


/*
扭曲块的部分区域
*/
#define   QUADRANT  4
#define   TWAIST_DEGREE  30
void twist_data(GTYPE *in, int row, int col, int chn, int start_row, int start_col, int s_row, int s_col, int w, int h)
{
	int end_row = start_row + h;
	int end_col = start_col + w;

	if (end_row > row - s_row) end_row = row - s_row;
	if (end_col > col - s_col) end_col = col - s_col;

	for (int i = start_row; i < end_row; i += s_row)
	{
		for (int j = start_col; j < end_col; j += s_col)
		{
			int pos = rand() % QUADRANT + 1;
			int twist_degree = rand() % (2 * TWAIST_DEGREE + 1) - TWAIST_DEGREE;
			twist(in, row, col, chn, i, j, s_row, s_col, pos, twist_degree);
		}
	}
}




void rotate_Image(GTYPE *buf, int src_row, int src_col, int degree)
{
	Mat src(src_row, src_col, CV_32FC1);
	Mat dest;
	Size s = Size(src_row, src_col);
	
	for (int i = 0; i < src_row; i++)
	{
		for (int j = 0; j < src_col; j++)
		{
			src.at<float>(i, j) = buf[i*src_col + j];
		}
	}

	IplImage img(src);
	CvPoint2D32f center;
	center.x = float(img.width / 2.0 + 0.5);
	center.y = float(img.height / 2.0 + 0.5);
	//计算二维旋转的仿射变换矩阵
	Mat M = getRotationMatrix2D(center, degree, 1);
	warpAffine(src, dest, M, s);
	for (int m = 0; m < src_row; m++)
	{
		for (int n = 0; n < src_col; n++)
		{
			buf[m*src_col + n] = dest.at<float>(m, n);
		}
	}
}

#define  TWIST_ROW 10
#define  TWIST_COL 10
void data_augm(GTYPE *in, int row, int col, int chn, int m_x, int m_y)
{
	int rot = rand() % 6;
	switch (rot)
	{
	case 0:
		//水平翻转
		hor_rot(in, row, col, chn);
		break;
	case 1:
		//垂直翻转
		ver_rot(in, row, col, chn);
		break;
	case 2:	//平移
	{
		int move_x = rand() % (m_x * 2 + 1) - m_x;
		int move_y = rand() % (m_y * 2 + 1) - m_y;
		move(in, row, col, chn, move_x, move_y);
	}
	break;
	case 3:	//小于30度的旋转
	{
		int rot_degree = rand() % 61  - 30;
		rotate(in, row, col, chn, rot_degree);
	}
	break;
	case 4:	//小于15度的扭曲
	{
		twist_data(in, row, col, chn, 0, 0, TWIST_ROW, TWIST_COL, row, col);
	}
	break;
	default:
		//给多点机会给原生态数据
		break;
	}
}



void create_avg_data(char *src_file, char *dst_file, char* avg_file, int datasize)
{
	ifstream in;
	in.open(src_file, ios::in | ios::binary);
	if (!in.is_open())
	{
		printf("open %s error\n", src_file);
		return;
	}

	ofstream out, avg;
	out.open(dst_file, ios::out | ios::trunc | ios::binary);
	if (!out.is_open())
	{
		printf("open file %s error!\n", dst_file);
		return;
	}

	avg.open(avg_file, ios::out | ios::trunc | ios::binary);
	if (!avg.is_open())
	{
		printf("open file %s error!\n", avg_file);
		return;
	}

	GTYPE *data = new GTYPE[datasize];
	GTYPE *ex = new GTYPE[datasize];

	memset(ex, 0, sizeof(GTYPE)*datasize);
	int count = 0;
	while (1)
	{
		in.read((char*)data, datasize * sizeof(GTYPE));
		if (in.eof()) break;
		count++;
		for (int i = 0; i < datasize; i++)
			ex[i] += data[i];
	}
	if (count == 0) return;
	for (int i = 0; i < datasize; i++)
		ex[i] = ex[i] / count;
	avg.write((char*)ex, sizeof(GTYPE)*datasize);
	avg.close();

	printf("avg over\n");

	in.clear();
	in.seekg(0, ios::beg);
	while (1)
	{
		in.read((char*)data, datasize * sizeof(GTYPE));
		if (in.eof()) break;
		for (int i = 0; i < datasize; i++)
			data[i] = data[i] - ex[i];
		out.write((char*)data, sizeof(GTYPE)*datasize);
	}
	in.close();
	out.close();
}

void change_avg_data(char *src_file, char *dst_file, char* avg_file, int datasize)
{
	ifstream in, avg;
	in.open(src_file, ios::in | ios::binary);
	if (!in.is_open())
	{
		printf("open %s error\n", src_file);
		return;
	}

	avg.open(avg_file, ios::in | ios::binary);
	if (!avg.is_open())
	{
		printf("open %s error\n", avg_file);
		return;
	}

	ofstream out;
	out.open(dst_file, ios::out | ios::trunc | ios::binary);
	if (!out.is_open())
	{
		printf("open file %s error!\n", dst_file);
		return;
	}

	GTYPE *data = new GTYPE[datasize];
	GTYPE *ex = new GTYPE[datasize];

	avg.read((char*)ex, datasize * sizeof(GTYPE));
	avg.close();

	while (1)
	{
		in.read((char*)data, datasize * sizeof(GTYPE));
		if (in.eof()) break;
		for (int i = 0; i < datasize; i++)
			data[i] = data[i] - ex[i];
		out.write((char*)data, sizeof(GTYPE)*datasize);
	}
	in.close();
	out.close();
}


void update_array(streamoff *src, int &len, int pos)
{	
	for (int i = pos; i < len - 1; i++)
	{
		src[i] = src[i + 1];
	}
	len = len - 1;
}

void rand_data(char *src_train, char *dst_train, int train_size, char* src_label, char* dst_label, int label_size)
{
	ifstream in_train, in_label;
	in_train.open(src_train, ios::in | ios::binary | ios::ate);
	if (!in_train.is_open())
	{
		printf("open %s error\n", src_train);
		return;
	}
	int  size = in_train.tellg() / (train_size * sizeof(GTYPE));
	in_train.seekg(0, ios::beg);

	in_label.open(src_label, ios::in | ios::binary );
	if (!in_label.is_open())
	{
		printf("open %s error\n", src_label);
		return;
	}

	streamoff *hash = new streamoff[size];
	for (int i = 0; i < size; i++)
		hash[i] = i;


	ofstream out_train,out_label;
	out_train.open(dst_train, ios::out | ios::trunc | ios::binary);
	if (!out_train.is_open())
	{
		printf("open file %s error!\n", dst_train);
		return;
	}
	out_label.open(dst_label, ios::out | ios::trunc | ios::binary);
	if (!out_label.is_open())
	{
		printf("open file %s error!\n", dst_label);
		return;
	}

	time_t now = time(NULL);
	srand(now);

	GTYPE *train_data = new GTYPE[train_size];
	GTYPE *label_data = new GTYPE[label_size];

	int count = 0;
	int  len = size;
	while (count < size)
	{
		int rand_pos = rand() % len;
		streamoff offset = hash[rand_pos] *train_size * sizeof(GTYPE);
		in_train.seekg(offset, ios::beg);
		in_train.read((char*)train_data, train_size * sizeof(GTYPE));
		offset = hash[rand_pos] * label_size * sizeof(GTYPE);
		in_label.seekg(offset, ios::beg);
		in_label.read((char*)label_data, label_size * sizeof(GTYPE));

		out_train.write((char*)train_data, sizeof(GTYPE)*train_size);
		out_label.write((char*)label_data, sizeof(GTYPE)*label_size);

		update_array(hash, len, rand_pos);		
		count++;
	}
	in_train.close();
	in_label.close();
	out_train.close();
	out_label.close();
}

void random_take_data(char *src_train_data, char *src_train_label, char *dst_train_data, char *dst_train_label, char *save_train, char *save_label,int data_size, int label_size, int num)
{
	ifstream in_train, in_label;
	in_train.open(src_train_data, ios::in | ios::binary | ios::ate);
	if (!in_train.is_open())
	{
		printf("open %s error\n", src_train_data);
		return;
	}
	int  size = in_train.tellg() / (data_size * sizeof(GTYPE));
	in_train.seekg(0, ios::beg);

	in_label.open(src_train_label, ios::in | ios::binary);
	if (!in_label.is_open())
	{
		printf("open %s error\n", src_train_label);
		return;
	}

	streamoff *hash = new streamoff[size];
	for (int i = 0; i < size; i++)
		hash[i] = i;

	ofstream out_train, out_label;
	out_train.open(dst_train_data, ios::out | ios::trunc | ios::binary);
	if (!out_train.is_open())
	{
		printf("open file %s error!\n", dst_train_data);
		return;
	}

	out_label.open(dst_train_label, ios::out | ios::trunc | ios::binary);
	if (!out_label.is_open())
	{
		printf("open file %s error!\n", dst_train_label);
		return;
	}

	ofstream save_tf, save_tl;
	save_tf.open(save_train, ios::out | ios::trunc | ios::binary);

	if (!save_tf.is_open())
	{
		printf("open file %s error!\n", save_train);
		return;
	}

	save_tl.open(save_label, ios::out | ios::trunc | ios::binary);
	if (!save_tl.is_open())
	{
		printf("open file %s error!\n", save_label);
		return;
	}

	time_t now = time(NULL);
	srand(now);

	GTYPE *train_data = new GTYPE[data_size];
	GTYPE *label_data = new GTYPE[label_size];

	int  len = size;
	for (int i = 0; i < num; i++)
	{
		int rand_pos = rand() % len;
		streamoff offset = hash[rand_pos] * data_size * sizeof(GTYPE);
		in_train.seekg(offset, ios::beg);
		in_train.read((char*)train_data, data_size * sizeof(GTYPE));
		offset = hash[rand_pos] * label_size * sizeof(GTYPE);
		in_label.seekg(offset, ios::beg);
		in_label.read((char*)label_data, label_size * sizeof(GTYPE));

		out_train.write((char*)train_data, sizeof(GTYPE)*data_size);
		out_label.write((char*)label_data, sizeof(GTYPE)*label_size);

		update_array(hash, len, rand_pos);
	}

	for (int i = 0; i < len; i++)
	{
		streamoff offset = hash[i] * data_size * sizeof(GTYPE);
		in_train.seekg(offset, ios::beg);
		in_train.read((char*)train_data, data_size * sizeof(GTYPE));

		offset = hash[i] * label_size * sizeof(GTYPE);
		in_label.seekg(offset, ios::beg);
		in_label.read((char*)label_data, label_size * sizeof(GTYPE));

		save_tf.write((char*)train_data, sizeof(GTYPE)*data_size);
		save_tl.write((char*)label_data, sizeof(GTYPE)*label_size);
	}

	in_train.close();
	in_label.close();
	out_train.close();
	out_label.close();
	save_tf.close();
	save_tl.close();
}


void data_rand_argu(char *src_train_data, char *src_train_label, char *dst_train_data, char *dst_train_label,int row, int col, int chn, int label_size, int num)
{
	int data_size = row * col * chn;

	ifstream in_train, in_label;
	in_train.open(src_train_data, ios::in | ios::binary | ios::ate);
	if (!in_train.is_open())
	{
		printf("open %s error\n", src_train_data);
		return;
	}
	
	in_label.open(src_train_label, ios::in | ios::binary);
	if (!in_label.is_open())
	{
		printf("open %s error\n", src_train_label);
		return;
	}

	ofstream out_train, out_label;
	out_train.open(dst_train_data, ios::out | ios::trunc | ios::binary);
	if (!out_train.is_open())
	{
		printf("open file %s error!\n", dst_train_data);
		return;
	}

	out_label.open(dst_train_label, ios::out | ios::trunc | ios::binary);
	if (!out_label.is_open())
	{
		printf("open file %s error!\n", dst_train_label);
		return;
	}

	GTYPE *train_data = new GTYPE[data_size];
	GTYPE *temp_data = new GTYPE[data_size];
	GTYPE *label_data = new GTYPE[label_size];

	time_t now = time(NULL);
	srand(now);

	while(1)
	{
		in_train.read((char*)train_data, data_size * sizeof(GTYPE));
		if (in_train.eof()) break;
		in_label.read((char*)label_data, label_size * sizeof(GTYPE));

		for (int i = 0; i < num; i++)
		{
			memcpy(temp_data, train_data, data_size * sizeof(GTYPE));
			data_augm(temp_data, row, col, chn, 4, 4);
			out_train.write((char*)temp_data, sizeof(GTYPE)*data_size);
			out_label.write((char*)label_data, sizeof(GTYPE)*label_size);
		}
	}
	in_train.close();
	in_label.close();
	out_train.close();
	out_label.close();
}

#endif