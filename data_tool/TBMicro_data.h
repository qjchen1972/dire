#pragma once
#ifndef _TBMICRO__H
#define _TBMICRO__H

#include<opencv2/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
#include<fstream>
#include<vector>
#include<set>
#include<time.h>
#include"tool_api/const.h"
#include "tool_api/possion.h"

#define  DATA_ROW  64
#define  DATA_COL  64
#define  TRAIN_DATA "data/train_data"
#define  TRAIN_LABEL  "data/train_label"
#define MARKING_TXT  "MarkingImg.txt"
#ifdef _DEBUG
#define SRC_IMAGE_DIR "D:/NewPlace/GenerateTrainData/x64/Debug/data/TBsrc"
#else
#define SRC_IMAGE_DIR    "srcdata"
#endif

#define SEPARATOR "/"
#define THUMB_IMAGE_DIR    "64x64" 
#define CUR_DIR     ""

#define TYPE_NUM 5
#define TB_SRC_ROW 1040
#define TB_SRC_COL 1376
#define TB_SRC_CHN 2
#define MAX_FINE_NUM 100  //���֧��100�Ŵ�ͼ
using namespace cv;
using namespace std;
map<int, vector<Rect> > g_TBRois; //��¼ÿ��ͼƬ��TB rois
Mat g_srcMatrixs_x[MAX_FINE_NUM];  //��¼ÿ��ͼƬ��X�ݶ�ͼ(��ͨ��)
Mat g_srcMatrixs_y[MAX_FINE_NUM];  //��¼ÿ��ͼƬ��Y�ݶ�ͼ(��ͨ��)
Mat g_srcMat[MAX_FINE_NUM]; //ԭͼ
int FILE_NUM = 20;

//ģ�庯������string���ͱ���ת��Ϊ���õ���ֵ���ͣ��˷��������ձ������ԣ�  
template <class Type> Type stringToNum(const string& str) 
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

//��ȡͼƬ���
int getImgIndex(const string &in_str)
{
	int index = in_str.find("(");
	string tempStr = in_str.substr(index + 1, 3);  //ԭͼ������100�ţ����ȡ3���ַ�
	int num = stringToNum<int>(tempStr);
	return num;
}

//��ȡ�ַ���
//str��Դ�ַ���
//pattern�� �ָ���
std::vector<std::string> splitWithStl(const std::string &str, const std::string &pattern)
{
	std::vector<std::string> resVec;
	if ("" == str)	return resVec;
	//�����ȡ���һ������
	std::string strs = str + pattern;
	size_t pos = strs.find(pattern);
	size_t size = strs.size();
	while (pos != std::string::npos)
	{
		std::string x = strs.substr(0, pos);
		resVec.push_back(x);
		strs = strs.substr(pos + 1, size);
		pos = strs.find(pattern);
	}
	return resVec;
}

//�����ַ�����ȡ������������
bool getRectfromStr(const string &in_str, Rect *out_Rect)
{
	//�ٴ�in_str��ȡx1, y1, x3, y3������
	int iStartPos = in_str.find('(');
	if (-1 == iStartPos)	return false;
	iStartPos += 1;
	int iEndPos = in_str.find(')');
	if (-1 == iEndPos)	return false;
	string sRectStr = in_str.substr(iStartPos, iEndPos - iStartPos);
	//(894,217,909,244) �ֳ��ĸ�����
	int num[4] = { 0 };
	vector<string> stringlist = splitWithStl(sRectStr, ",");
	for (int i = 0; i < 4; i++)
		num[i] = stringToNum<int>(stringlist[i]);	
	Point_<int> pt1(num[0], num[1]);
	Point_<int> pt2(num[2], num[3]);
	out_Rect->x = min(pt1.x, pt2.x);
	out_Rect->y = min(pt1.y, pt2.y);
	out_Rect->width = max(pt1.x, pt2.x) - out_Rect->x;
	out_Rect->height = max(pt1.y, pt2.y) - out_Rect->y;
	return true;
}

//��ȡͼƬ����
bool getImgROI(const string &in_str, int in_ImgIndex)
{
	//train (1).bmp:: (733,350,754,374);(722,366,746,387);(712,382,738,395);(1054,604,1074,625);
	int StrPos = in_str.find("::");
	if (-1 == StrPos)	return false;
	string tempStr = in_str.substr(StrPos, in_str.length() - StrPos);
	//v.reverse(80);
	//:: (733, 350, 754, 374); (722, 366, 746, 387); (712, 382, 738, 395); (1054, 604, 1074, 625);
	vector<string> stringlist = splitWithStl(tempStr, ";");
	int iTBcounts = stringlist.size();
	g_TBRois[in_ImgIndex].reserve(iTBcounts + 1);
	//printf("img(%d)iTBcounts= %d, capacity=%d\n", in_ImgIndex, iTBcounts, g_TBRois[in_ImgIndex].capacity());
	for (vector<string>::iterator iter = stringlist.begin(); iter != stringlist.end();	iter++)
	{		//(733, 350, 754, 374)
		string sss = *iter;
		Rect ROI;
		if (true == getRectfromStr(sss, &ROI))
		{
			g_TBRois[in_ImgIndex].push_back(ROI);
		//	printf("img(%d)(%d,%d,%d,%d)\n", in_ImgIndex, ROI.x, ROI.y, ROI.x + ROI.width, ROI.y + ROI.height);
		}
	}
	return true;
}

//�������ļ���ȡ�ѱ�ǵ�TB����
bool getTBRoi(char *img_path)
{	char sMarkFileName[128] = { 0 };
	printf("--------[IN]getTBRoi.--------\n"); //debug
	sprintf(sMarkFileName, "%s/%s", img_path, MARKING_TXT);
	ifstream ifs;
	ifs.open(sMarkFileName, ios::in);
	if (!ifs.is_open())
	{
		printf("open file %s error!\n", sMarkFileName);
		return false;
	}
	unsigned int count = 0;
	const int BUFF_SIZE = 10240;
	char buffer[BUFF_SIZE];
	while (!ifs.eof())
	{
		count++;
		memset(buffer, 0, BUFF_SIZE * sizeof(char));
		ifs.getline(buffer, BUFF_SIZE);
		if (buffer[0] == '\0')	continue;
		string curLine(buffer);
		int iImgIndex = getImgIndex(curLine);   //��ȡͼƬ���
		if (iImgIndex<0)		return false;
		//printf("curLine.size() = %d\n", curLine.size());
		if (false == getImgROI(curLine, iImgIndex))//��������Ϣ���浽map<��ţ�����>g_TBRois��
			return false;
	}
	printf("getTBRoi finished\n");
	return true;
}


/**
* @brief �����������ε��ཻ�������������ͬʱ�����ཻ���ռ�������ı���
* @param ��һ�����ε�λ��
* @param �ڶ������ε�λ��
* @param ���������ཻ�������С
* @param ����������ϵ������С
* @return ���������ཻ���ռ�������ı���,���غϱ��������������Ϊ0���򷵻�0
*/
float computRectJoinUnion(const Rect &rc1, const Rect &rc2, float& AJoin, float& AUnion)
{
	CvPoint p1, p2;                 //p1Ϊ�ཻλ�õ����Ͻ����꣬p2Ϊ�ཻλ�õ����½�����
	p1.x = max(rc1.x, rc2.x);
	p1.y = max(rc1.y, rc2.y);
	p2.x = min(rc1.x + rc1.width, rc2.x + rc2.width);
	p2.y = min(rc1.y + rc1.height, rc2.y + rc2.height);
	AJoin = 0;
	if (p2.x > p1.x && p2.y > p1.y)            //�ж��Ƿ��ཻ
	{
		AJoin = (p2.x - p1.x)*(p2.y - p1.y);    //����Ƚ�������ཻ���
	}
	float A1 = rc1.width * rc1.height;    //��һ����������(��64x64����ͼ�����)
	float A2 = rc2.width * rc2.height;    //�ڶ�����������(С���󣬼�TBroi)
	AUnion = (A1 + A2 - AJoin);                 //������ϵ����
	if (AUnion > 0)
		return (AJoin / A2);                  //�ཻ��� �� С������� �ı���
	else
		return 0;
}

//����64x64����ͼ����㣬���������ͼ�а����ĸ˾�����
//index: ԭͼͼƬ���  
//x, y: 64x64����ͼ���������
unsigned int getTBcounts(int index, int x, int y)
{
	//��ѯg_TBRois[index], �ж�ÿ���������򣬿����ļ���������64x64����ͼ�С�
	Size s(DATA_ROW, DATA_COL);
	Point_<int> p(x, y);
	Rect ImgRec(p, s);
	unsigned int iTBCounts = 0;
	for (vector<Rect>::iterator iter = g_TBRois[index].begin();iter != g_TBRois[index].end();	iter++)
	{		float AJoin = 0.0f;
		float AUnion = 0.0f;
		float  fProportion = computRectJoinUnion(ImgRec, *iter, AJoin, AUnion);
		if (fProportion > 0.5)//�ཻ��ռ�ȴ���0.5,�˾���+1
		{			iTBCounts++;
		}
		//****************************�����*************************
		// > 0.8 ֱ��iTBCounts++
		// 0.2~0.8֮�䣬�ж����ص��������50%�������ص���64x64���ݿ��в�iTBCounts++
		// < 0.2 ֱ��continue
		//****************************�����*************************
	}
	return iTBCounts;
}

//���srcdata�е�ԭͼ������ԭͼ(�Ҷ�ͼ)�������ݶ�ͼ��split����������ͨ��
bool getMatrix(char *img_path, int in_ImgNum = 1)
{	Mat src;
	Mat X;
	Mat Y;
	Possion pos;
	Mat rgbx_channel[3];
	Mat rgby_channel[3];
	
	memset(g_srcMatrixs_x, 0, sizeof(Mat)*MAX_FINE_NUM);
	memset(g_srcMatrixs_y, 0, sizeof(Mat)*MAX_FINE_NUM);
	char file_name[128] = { 0 };
	printf("--------[IN]getMatrix.--------\n"); //debug
	for (int i = 0; i < in_ImgNum; i++)
	{	
		sprintf(file_name, "%s/train (%d).bmp", img_path, i);
		src = imread(file_name);
		if (src.rows == 0 || src.cols == 0)
		{
			printf("load src failed: %s\n", file_name);
			return false;  //��ȡͼƬʧ��
		}

		pos.set_outchan(src.channels());
		pos.computeGradientX(src, X);
		pos.computeGradientY(src, Y);
		split(X, rgbx_channel);
		split(Y, rgby_channel);
		rgbx_channel[0].copyTo(g_srcMatrixs_x[i]);
		rgby_channel[0].copyTo(g_srcMatrixs_y[i]);
	}
	return true;
}

//��ȡָ��ĳ��ͼƬ���ݶ�ͼ
bool get_one_Matrix(char *img_name, int index)
{
	Mat src;
	Mat X;
	Mat Y;
	Possion pos;
	Mat rgbx_channel[3];
	Mat rgby_channel[3];
	src = imread(img_name);
	if (src.rows == 0 || src.cols == 0)
	{
		printf("load src failed: %s\n", img_name);
		return false;  //��ȡͼƬʧ��
	}
	pos.set_outchan(src.channels());
	pos.computeGradientX(src, X);
	pos.computeGradientY(src, Y);	
	split(X, rgbx_channel);
	split(Y, rgby_channel);
	rgbx_channel[0].copyTo(g_srcMatrixs_x[index]);
	rgby_channel[0].copyTo(g_srcMatrixs_y[index]);
	return true;
}

//�õ���ͨ�����ݶ�ͼ
void TB_get_data_from_gradimg_with_pos(GTYPE* out, int img_index, int row, int col, int pos_x, int pos_y, bool *isAllBlack = NULL)
{
	Mat X = g_srcMatrixs_x[img_index];   //ˮƽͼ�ݶ�ͼ
	Mat Y = g_srcMatrixs_y[img_index];   //��ֱ�ݶ�ͼ

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
				out[h * col + w] = X.at<float>(pos_x + h, pos_y + w);
				out[row*col + h * col + w] = Y.at<float>(pos_x + h, pos_y + w);

				if (!(out[h * col + w] < 0.00001 && out[h * col + w] > -0.00001)) //�����ص㲻�Ǻ�ɫ
				{
					*isAllBlack = false;
				}
			}
		}
}

//���srcdata�е�ԭͼ������ԭͼ(�Ҷ�ͼ)����������ͨ��
bool get_srcMatrix(char *img_path, int in_ImgNum = 1)
{	Mat src;
	Mat rgb_channel[3];


	char file_name[128] = { 0 };
	printf("--------[IN]getMatrix.--------\n"); //debug
	for (int i = 0; i < in_ImgNum; i++)
	{		sprintf(file_name, "%s/train (%d).bmp", img_path, i);
		src = imread(file_name);
		if (src.rows == 0 || src.cols == 0)
		{
			printf("load src failed: %s\n", file_name);
			return false;  //��ȡͼƬʧ��
		}

		split(src, rgb_channel);	
		rgb_channel[0].copyTo(g_srcMat[i]);
	}
	return true;
}

//��ȡָ��ĳ��ͼƬ
bool get_one_srcMat(char *img_name, int index)
{
	Mat src;
	Mat rgb_channel[3];

	src = imread(img_name);
	if (src.rows == 0 || src.cols == 0)
	{
		printf("load src failed: %s\n", img_name);
		return false;  //��ȡͼƬʧ��
	}	
	split(src, rgb_channel);
	rgb_channel[0].copyTo(g_srcMat[index]);
	return true;
}

//�õ���ͨ����ԭͼ
void TB_get_data_from_img_with_pos(GTYPE* out, int img_index, int row, int col, int pos_x, int pos_y, bool *isAllBlack = NULL)
{
	Mat src = g_srcMat[img_index];   //ԭ�и�ͼ

	for (int h = 0; h < row; h++)
		for (int w = 0; w < col; w++)
		{
			if (pos_x + h >= src.rows || pos_y + w >= src.cols)
			{
				out[h * col + w] = 0;
			}
			else
			{
				out[h * col + w] = src.at<unsigned char>(pos_x + h, pos_y + w);
				if(src.at<unsigned char>(pos_x + h, pos_y + w) != 255)  //�����ص㲻�ǰ�ɫ
					*isAllBlack = false;				
			}
		}
}


int TB_create_one_data(ofstream &df, ofstream &lf, unsigned int img_index = 0, int isNeedLabel = 0)
{
	unsigned int data_size = DATA_ROW*DATA_COL;
	GTYPE *value = new GTYPE[data_size];  //64x64��X��Y�����ݶ�ͼ���棬����X2
	GTYPE  ans[TYPE_NUM] = { 0 };
	bool isAllblack = true;
	unsigned int count = 0;
	int TBcounts[TYPE_NUM] = { 0 };
	//��1376x1040��ͼ��˳���ֳ�64x64Сͼ�������ϱ�ǩ
	for (int rows = 0; rows < TB_SRC_ROW; rows += DATA_ROW)
		for (int cols = 0; cols < TB_SRC_COL; cols += DATA_COL)
		{	
			isAllblack = true;
			//TB_get_data_from_gradimg_with_pos(value, img_index, DATA_ROW, DATA_COL, rows, cols, &isAllblack);
			TB_get_data_from_img_with_pos(value, img_index, DATA_ROW, DATA_COL, rows, cols, &isAllblack);
			if (true == isAllblack) //ɾ��ȫ�ڵ�ͼƬ
				continue;
			cout << rows << "  " << cols;
			count++;
			df.write((char*)value, sizeof(GTYPE)*data_size);
			if (isNeedLabel != 0)
			{
				unsigned int iTBCounts = getTBcounts(img_index, cols, rows);
				if (iTBCounts > TYPE_NUM - 1)	iTBCounts = TYPE_NUM - 1;
				TBcounts[iTBCounts]++;
				memset(ans, 0, sizeof(GTYPE) * TYPE_NUM);
				ans[iTBCounts] = 1;
			}
			lf.write((char*)ans, sizeof(GTYPE) * TYPE_NUM);
		}
	int iAllCounts = TBcounts[1] + 2 * TBcounts[2] + 3 * TBcounts[3] + 4 * TBcounts[4];
	printf("(%d)count0 = %d, count1 = %d, count2 = %d,count3 = %d, count4 = %d, All=%d\n", img_index, TBcounts[0], TBcounts[1], TBcounts[2], TBcounts[3], TBcounts[4], iAllCounts);	
	delete[] value;
	return count;
}


//������˸˾���ѵ������(�½ӿڣ���˳������ѵ������)
void TB_create_train_data(char* train_file, char* label_file, char* img_path, int src_num = 1)
{
	if (NULL == train_file || NULL == label_file || src_num <= 0)	return;
	FILE_NUM = src_num;
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
	if (false == getTBRoi(img_path))//��ȡԭͼ�е�TB������Ϣ
		return;
	//if (false == getMatrix(img_path, src_num))//��ȡ�ݶ�ͼ��Ϣ
		//return;

	if (false == get_srcMatrix(img_path, src_num))//��ȡԭͼ��Ϣ	
		return;
	unsigned int dataAmount = 0;
	for (unsigned int i = 0; i < src_num; i++)
	{
		dataAmount += TB_create_one_data(df, lf, i, 1);
	}
	df.close();
	lf.close();
	printf("create_train_data finished...\n dataAmount is %d\n", dataAmount);
}

/**
* @brief ������˸˾��ĵ�����������
* @param ����������и�ͼ�ĻҶ�ͼ����(����·��)
* @param ������������ɵ�datafile
* @param ������������ɵ�labelfile
*/
int TB_create_one_test(char *datafile, char* labelfile, char* img_path, int img_index = 0, int isNeedLabel = 1)
{
	ofstream df, lf;
	df.open(datafile, ios::out | ios::trunc | ios::binary);
	if (!df.is_open())
	{
		printf("open file %s error!\n", datafile);
		return -1;
	}	
	lf.open(labelfile, ios::out | ios::trunc | ios::binary);
	if (!lf.is_open())
	{
		printf("open file %s error!\n", labelfile);
		return -1;
	}
	if(1 == isNeedLabel)
		if (false == getTBRoi(img_path))//��ȡԭͼ�е�TB������Ϣ
			return -1;	
	char file_name[128] = { 0 };
	sprintf(file_name, "%s/train (%d).bmp", img_path, img_index);
	//if (false == get_one_Matrix(file_name, img_index))//��ȡԭͼ��Ϣ���������0�����0��ͼƬ"srcdata/train (0).bmp"
		//return -1;
	if (false == get_one_srcMat(file_name, img_index))//��ȡԭͼ��Ϣ���������0�����0��ͼƬ"srcdata/train (0).bmp"
		return -1;
	TB_create_one_data(df, lf, img_index, 1);
	printf("--------src create one test data finished.--------\n");
	lf.close();
	df.close();
}

//�������ɲ�������
void TB_create_test_data(char* test_data, char* test_label, char *img_path, int img_num = 1, int isNeedLabel = 0)
{
	ofstream df, lf;
	df.open(test_data, ios::out | ios::trunc | ios::binary);
	if (!df.is_open())
	{
		printf("open file %s error!\n", test_data);
		return;
	}
	lf.open(test_label, ios::out | ios::trunc | ios::binary);
	if (!lf.is_open())
	{		printf("open file %s error!\n", test_label);
		return ;
	}
	if (0 == img_num)
		return;
	else
		FILE_NUM = img_num;
	if (1 == isNeedLabel)
		if (false == getTBRoi(img_path))//��ȡԭͼ�е�TB������Ϣ
			return;
	//if (false == getMatrix(img_path, img_num))//��ȡԭͼ��Ϣ
		//return;	
	if (false == get_srcMatrix(img_path, img_num))//��ȡԭͼ��Ϣ
		return;
	for (int i = 0; i < img_num; i++)
	{
		TB_create_one_data(df, lf, i, isNeedLabel);
	}
	printf("TB create test data finished!\n");
	return;
}

//��ȡ�ļ���С
unsigned int getFileSize(ifstream &in_stream)
{
	streampos   pos = in_stream.tellg();     //save   current   position
	in_stream.seekg(0, ios::end);
	unsigned int ret = in_stream.tellg();
	in_stream.seekg(pos);     //restore   saved   position   
	return ret;
}

#define TEST_OUTPUT "D:/NewPlace/GenerateTrainData/x64/Debug/test_output"
#define TEXT_OUT "D:/NewPlace/GenerateTrainData/x64/Debug/outputdata.txt"
/**
* @brief �����Ա�test_output�ļ���datalabel�ļ������test_output��¼�ĸ˾���
* @param ���������test_output�ļ�·��
* @param ���������datalabel�ļ�·��
* @param �����������Ÿ˾�������BYTE�Ͷ����黺����
* @param ����������˾�������BYTE�Ͷ����黺�����Ĵ�С
*/
void TB_analyzeTestOutPut(char *in_testfile, char *in_datalabel = NULL,unsigned char *out_TBCounts = NULL, unsigned int in_iSize = 0)
{	ifstream ifs_testfile;
	ifs_testfile.open(in_testfile, ios::in | ios::binary);
	if (!ifs_testfile.is_open())
	{
		printf("analyzeTestOutPut open file %s error!\n", in_testfile);
		return;
	}
	ifstream ifs_datalabel;
	ifs_datalabel.open(in_datalabel, ios::in | ios::binary);
	if (!ifs_datalabel.is_open())
	{
		printf("analyzeTestOutPut open file %s error!\n", in_datalabel);
		return;
	}	
	unsigned int uiTest_size = getFileSize(ifs_testfile);
	unsigned int uiLabel_size = getFileSize(ifs_datalabel);
	unsigned int uiTest_num = uiTest_size / sizeof(GTYPE);
	unsigned int uiLabel_num = uiLabel_size / sizeof(GTYPE);
	GTYPE *test_buf = new GTYPE[uiTest_num + 1];
	GTYPE *label_buf = new GTYPE[uiLabel_num + 1];
	unsigned int read_size = TYPE_NUM * sizeof(GTYPE);
	if (NULL == test_buf || label_buf)
	{
		printf("new char[%d] error!\n", read_size);
		return;
	}
	//��ȡȫ���ļ���Ϣ�����뻺��
	ifs_testfile.read((char *)test_buf, uiTest_size);
	ifs_datalabel.read((char *)label_buf, uiLabel_size);
	ifs_testfile.close();
	ifs_datalabel.close();
	
	bool isCorrect = false;         //test_output�Ľ���Ƿ��label�Ľ��һ��
	float fCorrectRatio = 0.0f;     //test_output�Ľ����label�Ľ��һ�µĸ���
	unsigned int uiCorTimes = 0;    //test_output��label���һ�µ��ۼƴ���
	unsigned int totalCounts = 0;   //�ۼƸ˾�����
	unsigned int index_test, index_label;  //��¼������5��GTYPE����Ǹ������
	
	GTYPE maxVal_test, maxVal_label;       //��¼������5��GTYPE�е����ֵ
	for(unsigned int i = 0; i<uiTest_num /5 && i<uiLabel_num/5; i+=5)
	{
		maxVal_test = test_buf[i];
		maxVal_label = label_buf[i];
		index_test = 0;		index_label = 0;
		for (unsigned int j = i; j < i + 5;  j++)
		{
			if (maxVal_test < test_buf[i])
			{
				maxVal_test = test_buf[i];
				index_test = j - i;
			}
			if (maxVal_label < label_buf[i])
			{				maxVal_label = label_buf[i];
				index_label = j - i;
			}
			printf("%f(%f), ", test_buf[j], label_buf[j]);
		}
		totalCounts += index_test;
		isCorrect = maxVal_test == maxVal_label;
		if (true == isCorrect)
			uiCorTimes++;
		fCorrectRatio = uiCorTimes / i;
		printf("TBCounts:%d, %d, %.3f,\n", index_test, isCorrect, fCorrectRatio);
	}
	printf("total TB Counts: %d,  fCorrectRatio: %.3f\n", totalCounts, fCorrectRatio); //��Ҫ������ȷ��
	delete test_buf;
	delete label_buf;
}
#endif
