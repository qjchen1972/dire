#pragma once
#ifndef _IMG_PROC__H
#define _IMG_PROC__H

#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include "const.h"
#include "possion.h"
using namespace cv;
using namespace std;



struct TagValue
{
	unsigned short tag1;
	unsigned short tag2;
};

void read_dicom(const char* filename, Mat &src)
{
	bool isVR = true;
	bool isLitteEndian = true;
	int file_length = 0;
	char VR[3];

	unsigned int pixDataLen = 0;
	unsigned int pixDataOffset = 0;
	unsigned short channle = 0;
	unsigned short rows = 0;
	unsigned short cols = 0;
	unsigned short dataLen = 0;
	unsigned short validLen = 0;
	int windowsWidth = 0;
	int windowsCenter = 0;
	bool ZeroIsBlack = true;
	int RescaleSlope = 1;
	int RescaleIntercept = 0;

	ifstream f;
	f.open(filename, ios::in | ios::binary | ios::ate);
	if (!f.is_open())
	{
		printf("open %s error\n", filename);
		return ;
	}
	file_length = (int)f.tellg();
	f.seekg(128, ios::beg);

	char headchar[5];

	memset(headchar, 0, 5);
	f.read(headchar, 4);	
	if (strcmp(headchar, "DICM"))
	{
		f.close();
		return;
	}

	int count = 132;
	memset(VR, 0, 3);

#define  LEN  1024
	char msg[LEN];

	while ( count + 6 < file_length)
	{
		TagValue tag;
		unsigned int len;
		unsigned short slen;

		f.read((char*)&tag, sizeof(TagValue));
		count += sizeof(TagValue);

		if (tag.tag1 == 0x02)
		{
			f.read(VR, 2);
			count += 2;
			if (!strcmp(VR, "OB") || !strcmp(VR, "OW") || !strcmp(VR, "SQ") || !strcmp(VR, "OF") || !strcmp(VR, "UT") || !strcmp(VR, "UN"))
			{
				f.seekg(2, ios::cur);
				f.read((char*)&len, sizeof(unsigned int));
				count += len + 2;
			}
			else
			{
				f.read((char*)&slen, sizeof(unsigned short));
				len = slen;
				count += len;
			}
		}
		else if (tag.tag1 == 0xfffe)
		{
			if (tag.tag2 == 0xe000 || tag.tag2 == 0xe00d || tag.tag2 == 0xe0dd)
			{
				f.read((char*)&len, sizeof(unsigned int));
				count += len;
			}
		}
		else if (isVR == true)
		{
			f.read(VR, 2);
			count += 2;
			if (!strcmp(VR, "OB") || !strcmp(VR, "OW") || !strcmp(VR, "SQ") || !strcmp(VR, "OF") || !strcmp(VR, "UT") || !strcmp(VR, "UN"))
			{
				f.seekg(2, ios::cur);
				f.read((char*)&len, sizeof(unsigned int));
				count += len + 2;
			}
			else
			{
				f.read((char*)&slen, sizeof(unsigned short));
				len = slen;
				count += len;
			}
		}
		else if (isVR == false)
		{
			f.read((char*)&len, sizeof(unsigned int));
			count += len;
		}

		if (tag.tag1 == 0x02 && tag.tag2 == 0x10)
		{			
			memset(msg, 0, LEN);
			f.read(msg, len);
			count += len;
			if (!strcmp(msg, "1.2.840.10008.1.2.1"))
			{
				isLitteEndian = true;
				isVR = true;
			}
			else if (!strcmp(msg, "1.2.840.10008.1.2.2"))
			{
				isLitteEndian = false;
				isVR = true;
			}
			else if (!strcmp(msg, "1.2.840.10008.1.2"))
			{
				isLitteEndian = true;
				isVR = false;
			}
		}
		else if (tag.tag1 == 0x7fe0 && tag.tag2 == 0x10)
		{
			pixDataLen = len;
			pixDataOffset = (int)f.tellg();
			f.seekg(len, ios::cur);
			count += len;
		}
		else if (tag.tag1 == 0x28 && tag.tag2 == 0x10 && rows == 0)
		{
			f.read((char*)&rows, sizeof(unsigned short));
			count += 2;
		}
		else if (tag.tag1 == 0x28 && tag.tag2 == 0x11 && cols == 0)
		{
			f.read((char*)&cols, sizeof(unsigned short));
			count += 2;
		}
		else if (tag.tag1 == 0x28 && tag.tag2 == 0x02 && channle == 0)
		{
			f.read((char*)&channle, sizeof(unsigned short));
			count += 2;
		}
		else if (tag.tag1 == 0x28 && tag.tag2 == 0x101 && validLen == 0)
		{
			f.read((char*)&validLen, sizeof(unsigned short));
			count += 2;
		}
		else if (tag.tag1 == 0x28 && tag.tag2 == 0x100 && dataLen == 0)
		{
			f.read((char*)&dataLen, sizeof(unsigned short));
			count += 2;
		}
		else if (tag.tag1 == 0x28 && tag.tag2 == 0x1050 && windowsCenter == 0)
		{
			memset(msg, 0, LEN);
			f.read(msg, len);
			count += len;
			windowsCenter = atoi(msg);
		}
		else if (tag.tag1 == 0x28, tag.tag2 == 0x1051 && windowsWidth == 0)
		{
			memset(msg, 0, LEN);
			f.read(msg, len);
			count += len;
			windowsWidth = atoi(msg);
		}
		else if (tag.tag1 == 0x0028 && tag.tag2 == 0x0004)
		{
			memset(msg, 0, LEN);
			f.read(msg, len);
			count += len;
			if (!strcmp(msg, "MONOCHROME1 "))
			{
				ZeroIsBlack = false;
			}
			else if (!strcmp(msg, "MONOCHROME2 "))
			{
				ZeroIsBlack = true;
			}
		}
		else if (tag.tag1 == 0x0028 && tag.tag2 == 0x1052 && RescaleIntercept == 0)
		{
			memset(msg, 0, LEN);
			f.read(msg, len);
			count += len;
			RescaleIntercept = atoi(msg);
		}
		else if (tag.tag1 == 0x0028 && tag.tag2 == 0x1053 && RescaleSlope == 0)
		{
			memset(msg, 0, LEN);
			f.read(msg, len);
			count += len;
			RescaleSlope = atoi(msg);
		}
		else if (len == 0xffffffff || len == 0)
		{

		}
		else
		{
			f.seekg(len, ios::cur);
		}
	}

	f.seekg(pixDataOffset, ios::beg);

	if (windowsCenter == 0 && windowsWidth == 0)
	{
		windowsWidth = 1 << validLen;
		windowsCenter = windowsWidth / 2;
	}

	int min_value, max_value;
	min_value = windowsCenter - windowsWidth / 2.0 + 0.5;
	max_value = windowsCenter + windowsWidth / 2.0 + 0.5;
	double pers = 255.0 / (max_value - min_value);


	if (channle == 1)
	{
		src.create((int)rows, (int)cols, CV_8UC1);
		short gray = 0;
		unsigned char pix[2];

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				f.read((char*)pix, 2);
				if (validLen <= 8)
				{
					if (isLitteEndian)
					{
						gray = pix[0];
					}
					else
					{
						gray = pix[1];
					}
				}
				else
				{
					if (isLitteEndian)
					{
						gray = *(short*)pix;
						//if(RescaleSlope != 0)
						gray = gray * RescaleSlope + RescaleIntercept;
					}
					else
					{
						gray = pix[1] + pix[0] * 256;
					}

					if (gray < min_value)
						gray = 0;
					else if (gray > max_value)
						gray = 0xff;
					else
						gray = (int)((gray - min_value) * pers);
				}
				if (!ZeroIsBlack)
				{
					gray = 255 - gray;
				}
				src.at<uchar>(i, j) = gray;
			}
		}
	}
	else if (channle == 3)
	{
		unsigned char pix[3];
		src.create((int)rows, (int)cols, CV_8UC3);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				f.read((char*)pix, 3);
				src.at<Vec3b>(i, j)[0] = pix[2];
				src.at<Vec3b>(i, j)[1] = pix[1];
				src.at<Vec3b>(i, j)[2] = pix[0];
			}
		}
	}
	f.close();

	//imshow("src", src);
	//waitKey(0);
	return;
}

int create_bmp(char* img, int w, int h, int len, char *name)
{
	ofstream file(name, ios::out | ios::trunc | ios::binary);

#define BMP_Header_Length 54   
	char header[BMP_Header_Length] = {
		0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
		54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0
	};

	int w1;
	int mod = (3 * w) % 4;
	if (mod > 0) mod = 4 - mod;

	int file_size = h*(w * 3 + mod) + 54;
	header[2] = (file_size & 0x000000ff);
	header[3] = (file_size >> 8) & 0x000000ff;
	header[4] = (file_size >> 16) & 0x000000ff;
	header[5] = (file_size >> 24) & 0x000000ff;

	int width = w;
	header[18] = width & 0x000000ff;
	header[19] = (width >> 8) & 0x000000ff;
	header[20] = (width >> 16) & 0x000000ff;
	header[21] = (width >> 24) & 0x000000ff;

	int height = h;
	header[22] = height & 0x000000ff;
	header[23] = (height >> 8) & 0x000000ff;
	header[24] = (height >> 16) & 0x000000ff;
	header[25] = (height >> 24) & 0x000000ff;

	file.write(header, BMP_Header_Length);

	int type = len / (w * h);

	for (int i = h - 1; i >= 0; i--)
	{
		for (int j = 0; j < w; j++)
		{
			if (type == 1)
			{
				file.put(img[i*w + j]);
				file.put(img[i*w + j]);
				file.put(img[i*w + j]);
			}
			else
			{
				file.put(img[i*w + j]);
				file.put(img[w*h + i*w + j]);
				file.put(img[2 * w*h + i*w + j]);
			}
		}
		for (int k = 0; k < mod; k++)
		{
			file.put(0);
		}
	}
	file.close();
	return 1;
}

void get_data_from_gradimg(GTYPE* out, char *file, int row, int col)
{
	Mat  src;
	Size s = Size(row, col);
	src = imread(file);
	Mat new_src;

	int chn = src.channels();
	if (chn == 1)
		new_src = Mat(s, CV_8UC1);
	else
		new_src = Mat(s, CV_8UC3);
	resize(src, new_src, s);

	Possion pos;
	pos.set_outchan(chn);

	Mat X;
	pos.computeGradientX(new_src, X);	
	Mat Y;
	pos.computeGradientY(new_src, Y);
	
	if (chn == 1)
	{
		for (int h = 0; h < new_src.rows; h++)
			for (int w = 0; w < new_src.cols; w++)
			{
				out[h * new_src.cols + w] = X.at<float>(h, w);
				out[new_src.rows*new_src.cols+h * new_src.cols + w] = Y.at<float>(h, w);
			}
	}
	else
	{
		Mat rgbx_channel[3];
		Mat rgby_channel[3];

		split(X, rgbx_channel);
		split(Y, rgby_channel);

		for (int h = 0; h < new_src.rows; h++)
			for (int w = 0; w < new_src.cols; w++)
			{
				out[h * new_src.cols + w] = rgbx_channel[0].at<float>(h, w);
				out[new_src.rows*new_src.cols + h * new_src.cols + w] = rgby_channel[0].at<float>(h, w);
			}
	}
}

//得到单通道的梯度图
void get_data_from_gradimg_with_pos(GTYPE* out, char *file, int row, int col, int pos_x, int pos_y)
{
	Mat  src;
	src = imread(file);

	Possion pos;
	pos.set_outchan(src.channels());

	Mat X;
	pos.computeGradientX(src, X);
	Mat Y;
	pos.computeGradientY(src, Y);

	Mat rgbx_channel[3];
	Mat rgby_channel[3];

	split(X, rgbx_channel);
	split(Y, rgby_channel);

	for (int h = 0; h < row; h++)
		for (int w = 0; w < col; w++)
		{
			if (pos_x + h >= X.rows || pos_y + w >= X.cols)
			{
				out[h * col + w] = 0;
				out[row*col + h * col + w] = 0;
			}
			else
			{
				out[h * col + w] = rgbx_channel[0].at<float>(pos_x + h, pos_y + w);
				out[row*col + h * col + w] = rgby_channel[0].at<float>(pos_x + h, pos_y + w);
			}
		}
}
	



void get_data_from_img(GTYPE* out, char *file,int row, int col)
{
	Mat  src;
	Size s = Size(row, col);
	src = imread(file);
	Mat new_src;
	if(src.channels() == 1 )
		new_src = Mat(s, CV_8UC1);
	else
		new_src = Mat(s, CV_8UC3);
	resize(src, new_src, s);
	for (int h = 0; h < new_src.rows; h++)
	{
		unsigned char* s = new_src.ptr<unsigned char>(h);
		int img_index = 0;
		for (int w = 0; w < new_src.cols; w++)
		{
			for (int c = 0; c < new_src.channels(); c++)
			{
				int data_index = (c*new_src.rows + h)*new_src.cols + w;
				unsigned char g = s[img_index++];
				out[data_index] = g;
			}
		}
	}
}



//增加对比度
void add_light(Mat &src, int light)
{
	for (int m = 0; m < src.rows; m++)
		for (int n = 0; n < src.cols; n++)
		{
			if (m <= 1 || n <= 1 || m >= src.rows - 2 || n >= src.cols - 2)
			{
				if (src.at<uchar>(m, n) < 255 - light)
					src.at<uchar>(m, n) += light;
			}
		}
}

#endif
