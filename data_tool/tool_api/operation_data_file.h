#ifndef _OPERATION_DATA_FILE__H
#define _OPERATION_DATA_FILE__H

#include <iostream>
#include <fstream>
#include<string.h>
#include <iomanip>
#include "const.h"
using namespace std;

void CombinationData(char * lableFile, char * dataFile, char * combinationFile, int lableSize, int dataSize);
void CutFile(char * srcFile, char * destFile, int size);
void transData(char * srcFile, char * destFile, char * srcType, char * destType);


void CombinationData(char * lableFile, char * dataFile, char * combinationFile, int lableSize, int dataSize)
{
	ifstream  collectionData1;
	collectionData1.open(lableFile, ios::in | ios::binary);
	if (!collectionData1.is_open())
	{
		printf("open %s error\n", lableFile);
		return;
	}
	ifstream  collectionData2;
	collectionData2.open(dataFile, ios::in | ios::binary);
	if (!collectionData2.is_open())
	{
		printf("open %s error\n", dataFile);
		return;
	}

	ofstream combination;
	combination.open(combinationFile, ios::out | ios::binary);
	if (!combination.is_open())
	{
		printf("open file %s error!\n", combinationFile);
		return;
	}

	char *data1 = new char[lableSize];
	char *data2 = new char[dataSize];

	while (1)
	{
		collectionData1.read((char*)data1, lableSize);

		if (collectionData1.eof()) break;

		combination.write((char*)data1, lableSize);

		collectionData2.read((char*)data2, dataSize);

		if (collectionData2.eof()) break;

		combination.write((char*)data2, dataSize);

	}

	delete[] data1;
	delete[] data2;
	collectionData1.close();
	collectionData2.close();
	combination.close();
}


void CutFile(char * srcFile, char * destFile, int size)
{
	char *buf = new char[size];

	ifstream src;
	src.open(srcFile, ios::in | ios::binary);
	if (!src.is_open())
	{
		printf("open file %s error!\n", srcFile);
		return;
	}

	ofstream dest;
	dest.open(destFile, ios::out | ios::binary);

	if (!dest.is_open())
	{
		printf("open file %s error!\n", destFile);
		return;
	}

	src.read((char*)buf, size);
	dest.write((char *)buf, size);
	src.close();
	dest.close();
	delete[] buf;
}

void transData(char * srcFile, char * destFile, char * srcType, char * destType)
{
	ifstream  SrcConvertData;
	SrcConvertData.open(srcFile, ios::in | ios::binary);

	if (!SrcConvertData.is_open())
	{
		printf("open %s error\n", srcFile);
		return;
	}

	ofstream DestConvertData;
	DestConvertData.open(destFile, ios::out | ios::binary);

	if (!DestConvertData.is_open())
	{
		printf("open file %s error!\n", destFile);
		return;
	}

	if (!strcmp("float", srcType) && !strcmp("int", destType))
	{
		char *srcBuf = new char[sizeof(float)];
		float value = 0;
		int data = 0;

		while (1)
		{
			SrcConvertData.read((char*)srcBuf, sizeof(float));
			memcpy(&value, srcBuf, sizeof(float));
			data = value;
			if (SrcConvertData.eof()) break;
			DestConvertData.write((char *)&data, sizeof(int));
		}
		delete[] srcBuf;
	}

	else if (!strcmp("double", srcType) && !strcmp("float", destType))
	{
		char *srcBuf = new char[sizeof(double)];
		double value = 0;
		float data = 0;

		while (1)
		{
			SrcConvertData.read((char*)srcBuf, sizeof(double));
			memcpy(&value, srcBuf, sizeof(double));
			data = value;
			if (SrcConvertData.eof()) break;
			DestConvertData.write((char *)&data, sizeof(float));
		}
		delete[] srcBuf;
	}

	else if (!strcmp("float", srcType) && !strcmp("double", destType))
	{
		char *srcBuf = new char[sizeof(float)];
		float value = 0;
		double data = 0;

		while (1)
		{
			SrcConvertData.read((char*)srcBuf, sizeof(float));
			memcpy(&value, srcBuf, sizeof(float));
			data = value;
			if (SrcConvertData.eof()) break;
			DestConvertData.write((char *)&data, sizeof(double));
		}
		delete[] srcBuf;
	}

	else if (!strcmp("int", srcType) && !strcmp("float", destType))
	{
		char *srcBuf = new char[sizeof(int)];
		int value = 0;
		float data = 0;

		while (1)
		{
			SrcConvertData.read((char*)srcBuf, sizeof(int));
			memcpy(&value, srcBuf, sizeof(int));
			data = value;
			if (SrcConvertData.eof()) break;
			DestConvertData.write((char *)&data, sizeof(float));
		}

		delete[] srcBuf;
	}

	else if (!strcmp("int", srcType) && !strcmp("char", destType))
	{
		char *srcBuf = new char[sizeof(int)];
		int value = 0;
		char *data = new char[sizeof(char)];
		while (1)
		{
			SrcConvertData.read((char*)srcBuf, sizeof(int));
			memcpy(&value, srcBuf, sizeof(int));
			*data = value;
			if (SrcConvertData.eof()) break;
			DestConvertData.write((char *)&data, sizeof(char));
		}
		delete[] srcBuf;
		delete[] data;
	}

	else if (!strcmp("int", srcType) && !strcmp("short", destType))
	{
		char *srcBuf = new char[sizeof(int)];
		int value = 0;
		short data = 0;
		while (1)
		{
			SrcConvertData.read((char*)srcBuf, sizeof(int));
			memcpy(&value, srcBuf, sizeof(int));
			data = value;
			if (SrcConvertData.eof()) break;
			DestConvertData.write((char *)&data, sizeof(short));
		}
		delete[] srcBuf;
	}

	SrcConvertData.close();
	DestConvertData.close();
}

#endif