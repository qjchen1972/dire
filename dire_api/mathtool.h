/*
* Copyright (c) 2017
* All rights reserved.
*
* �ļ�����: mathtool.h
* �ļ���ʶ:
* ժ    Ҫ: һЩ���ߺ���
*
*
* ��ǰ�汾: 1.0
* ��    ��: chen.qian.jiang
* ��ʼ����: 2017-06-16
* �������:
* ����˵����
* �޸�����        �汾��     �޸���          �޸�����
* -----------------------------------------------
* 2017/06/16      V1.0       ��ǭ��          ����
*/
#pragma once
#ifndef _MATHTOOL__H
#define _MATHTOOL__H

#define _USE_MATH_DEFINES

#include "const.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include<string.h>
#include<stdio.h>
#include<chrono>
#include"setup.h"

using namespace std;
using namespace std::chrono;

#define BASE_COLOR  255

class Mathtool
{
public:
	Mathtool(){}
	~Mathtool() {}
	
	// ȡ�ø�˹�ֲ��������
	GTYPE gauss_rand(GTYPE ex, GTYPE var)
	{
		static GTYPE V1, V2, S;
		static int phase = 0;
		GTYPE X;

		if (phase == 0) {
			do {
				GTYPE U1 = (GTYPE)rand() / RAND_MAX;
				GTYPE U2 = (GTYPE)rand() / RAND_MAX;

				V1 = 2 * U1 - 1;
				V2 = 2 * U2 - 1;
				S = V1 * V1 + V2 * V2;

			} while (S >= 1 || S == 0);
			X = V1 * sqrt(-2 * log(S) / S);
		}
		else
			X = V2 * sqrt(-2 * log(S) / S);

		phase = 1 - phase;
		return sqrt(var) * X + ex;
	}

	// ȡ�þ��ȷֲ��������
	GTYPE uniform_rand(GTYPE dMinValue,GTYPE dMaxValue)
	{
		GTYPE pRandomValue = (GTYPE)(rand() / (GTYPE)RAND_MAX);
		pRandomValue = pRandomValue*(dMaxValue - dMinValue) + dMinValue;
		return pRandomValue;	
	}
	//��in�ļ�����Ӳ�̵��ڴ�
	bool open_in_file(ifstream &f, const char* file)
	{
		f.open(file, ios::in);		
		if (!f.is_open())
		{
			printf("open file %s error!\n", file);
			return false;
		}
		return true;
	}
	//��out�ļ������ڴ浽Ӳ��
	bool open_out_file(ofstream &f, const char* file,int type =0)
	{
		if( type == 0)
			f.open(file, ios::out | ios::trunc);
		else
			f.open(file, ios::out | ios::app);
		if (!f.is_open())
		{
			printf("open file %s error!\n", file);
			return false;
		}
		return true;
	}	

	/* �������ݶ�����
	x ����ǿ���ˮƽ����
	y ����ǿ��Ĵ�ֱ����
	sx ����ǿǰ��ˮƽ����
	sy ����ǿǰ�Ĵ�ֱ����
	*/

	/*
	ˮƽ��ת
	sx = (col-1) - x
	sy = y
	*/
	void hor_rot(GTYPE *in, int row, int col, int chn)
	{
		unsigned int node_size = row * col;
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
	��ֱ��ת
	sx = x
	sy = (row-1) - y
	*/
	void ver_rot(GTYPE *in, int row, int col, int chn)
	{
		unsigned int node_size = row * col;
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
	ƽ��
	sx = x + dx
	sy = y + dy
	*/
	void move(GTYPE *in, int row, int col, int chn, int move_x, int move_y)
	{
		unsigned int node_size = row * col;
		unsigned int batch_size = node_size*chn;
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
					in[k * node_size + j] = BASE_COLOR;
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
	��ת
	sx = x*cosa + y*sina //ˮƽ
	sy = y*cosa - x*sina //��ֱ
	*/
	void rotate(GTYPE *in, int row, int col, int chn, int degree)
	{
		GTYPE angle = degree * M_PI / 180.0;
		unsigned int node_size = row * col;
		unsigned int batch_size = node_size*chn;

		GTYPE *tempf = new GTYPE[batch_size];
		memcpy(tempf, in, batch_size * sizeof(GTYPE));

		int center_x =  col % 2 == 1 ? col / 2 + 1 : col / 2;
		int center_y = row % 2 == 1 ? row / 2 + 1 : row / 2;

		for (unsigned int j = 0; j < node_size; j++)
		{
			int node_x = j % col; //ˮƽ
			int node_y = j / col; //��ֱ			
			for (int k = 0; k < chn; k++)
			{
				int src_x = (node_x - center_x)  * cos(angle) + (node_y - center_y)  * sin(angle) + center_x;//ˮƽ
				int src_y = (node_y - center_y) * cos(angle) - (node_x - center_x)* sin(angle) + center_y;//��ֱ

				if (src_x < 0 || src_y < 0 || src_y >= row || src_x >= col)
				{
					in[k * node_size + j] = BASE_COLOR;
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
	Ť�������ĳһ����
	pos ������
	degree : �Ƕ�
	*/
	void twist(GTYPE *in, int row, int col, int chn, int pos, int degree)
	{
		unsigned int node_size = row * col;	
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
	Ť�������ĳһС���ĳһ����
	pos ������
	degree : �Ƕ�
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
	Ť����Ĳ�������
	*/
#define   QUADRANT  4
#define   TWAIST_DEGREE  30
	void twist_data(GTYPE *in, int row, int col, int chn, int start_row, int start_col,int s_row,int s_col,int w, int h)
	{
		int end_row = start_row + h;
		int end_col = start_col + w;

		if (end_row > row - s_row) end_row = row - s_row;
		if (end_col > col - s_col) end_col = col - s_col;
		
		for (int i = start_row; i < end_row; i += s_row )
		{
			for (int j = start_col; j < end_col; j += s_col )
			{
				int pos = rand() % QUADRANT + 1;
				int twist_degree = rand() % (2* TWAIST_DEGREE+1) - TWAIST_DEGREE;
				twist(in, row, col, chn, i, j, s_row, s_col, pos, twist_degree);
			}
		}
	}

private:

};
#endif