#include "lung_data.h"
#include "TBMicro_data.h"
#include "tool_api/data_proc.h"
#include "bone.h"
#include "tool_api/operation_data_file.h"


int main(int argc, char** argv)
{	
	if (argc < 2)
	{
		printf("Usage:  mode \n");
		printf("0 --- create average data: srcfile dstfile  avgfile size \n");
		printf("1 --- change data to average: srcfile dstfile avgfile size\n");
		printf("2 --- random data and label: src_trainfile dst_train train_size src_labelfile dst_labelfile lable_size\n");
		printf("3 --- create lung data: trainfile, labelfile, sick_num\n");
		printf("4 --- create one lung test data: srcfile, dstfile,labelfile, sick_type\n");
		printf("5 --- create lung test data: dstfile,labelfile, sick_num\n");
		printf("6 --- create TB data:train_file, label_file, img_path, src_img_num\n");
		printf("7 --- create one TB test data:datafile, labelfile, img_path, img_index\n");
		printf("8 --- TB test data:test_data, test_label, img_path, img_num \n");
		printf("9 --- print TB counts: test_output, test_label\n");
		printf("10 --- combination data file: labelfile,datafile,dstfile,label_size,data_size\n");
		printf("11 --- transform data type: srcfile,dstfile,srcDataType,destDataType\n");
		printf("13 --- create 14 kinds of lung train data:dst_train_data,dst_train_label \n");
		printf("20 --- create bone train data: trainfile, labelfile, total_row,total_col,total_num\n");
		printf("21 --- create one bone test data: srcfile, nobonefile, total_row, total_col, row, col, out_testfile, out_labelfile\n");
		printf("22 --- create bone test img: srcfile,gradfile,total_row, total_col, row, col, jpgfile, light\n");
		printf("30 --- random get data form file: srcfile,srclabel,dstfile,destlabel, savefile,savelabel,trainsize, labelsize, num\n");
		return 0;
	}

	int mode = atoi(argv[1]);

	if (mode == 0)
	{
		//产生均值文件
		int size = atoi(argv[5]);
		create_avg_data(argv[2], argv[3], argv[4], size);
	}
	else if (mode == 1)
	{
		//用均值文件产生数据
		int size = atoi(argv[5]);
		change_avg_data(argv[2], argv[3], argv[4], size);
	}
	else if (mode == 2)
	{
		//随机打乱数据文件及其相应的标签文件
		int train_size = atoi(argv[4]);
		int label_size = atoi(argv[7]);
		rand_data(argv[2], argv[3], train_size, argv[5], argv[6], label_size);
	}
	else if (mode == 3)
	{
		//产生肺病的训练数据
		int sick_num = atoi(argv[4]);
		lung_create_traing_data(argv[2], argv[3], sick_num);
	}
	else if (mode == 4)
	{
		//产生肺病的单个测试数据
		int type = atoi(argv[5]);
		lung_create_one_test(argv[2], argv[3], argv[4], type);
	}
	else if (mode == 5)
	{
		//产生肺病的测试数据
		int sick_num = atoi(argv[4]);
		lung_create_test_data(argv[2], argv[3], sick_num);
	}
	else if (mode == 6)
	{
		//产生结核杆菌的训练数据
		int src_num = atoi(argv[5]);		
		TB_create_train_data(argv[2], argv[3], argv[4], src_num);
	}
	else if (mode == 7)
	{
		//产生结核杆菌的单个测试数据
		int img_index = atoi(argv[5]);
		TB_create_one_test(argv[2], argv[3], argv[4], img_index, 1);
	}
	else if (mode == 8)
	{
		//产生结核杆菌的批量测试数据
		int img_num = atoi(argv[5]);
		//int isNeedLabel = atoi(argv[6]);
		TB_create_test_data(argv[2], argv[3], argv[4], img_num, 1/*, isNeedLabel*/);
	}
	else if (mode == 9)
	{
		//输入testoutput，分析杆菌数量
		TB_analyzeTestOutPut(argv[2], argv[3]);
	}
	else if (mode == 10)
	{
		//拼接标签与数据文件
		int lableSize = atoi(argv[5]);
		int dataSize = atoi(argv[6]);
		CombinationData(argv[2], argv[3], argv[4], lableSize, dataSize);
	}
	else if (mode == 11)
	{
		//更改数据类型
		transData(argv[2], argv[3], argv[4], argv[5]);
	}
	else if (mode == 13)
	{
		//产生14种肺病的训练数据
		creat_different_lung_train_data_src(argv[2], argv[3]);
	}
	else if (mode == 20)
	{
		//产生去骨训练数据
		int total_row = atoi(argv[4]);
		int total_col = atoi(argv[5]);
		int total_num = atoi(argv[6]);
		bone_create_train_data(argv[2], argv[3], total_row, total_col, total_num);
	}
	else if (mode == 21)
	{
		//产生去骨的测试数据
		int total_row = atoi(argv[4]);
		int total_col = atoi(argv[5]);
		int row = atoi(argv[6]);
		int col = atoi(argv[7]);
		bone_create_one_test(argv[2], argv[3], total_row, total_col, row, col, argv[8], argv[9]);
	}
	else if (mode == 22)
	{
		//产生去骨的测试图像
		int total_row = atoi(argv[4]);
		int total_col = atoi(argv[5]);
		int row = atoi(argv[6]);
		int col = atoi(argv[7]);
		int light = atoi(argv[9]);
		bone_create_one_test_img(argv[2], argv[3], total_row, total_col, row, col, argv[8], light);
	}
	else if (mode == 30)
	{
		//随机打乱数据文件及其相应的标签文件
		int train_size = atoi(argv[8]);
		int label_size = atoi(argv[9]);
		int num = atoi(argv[10]);
		random_take_data(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], train_size,label_size,num);
	}
	else if (mode == 31)
	{
		//数据增强

		int row = atoi(argv[6]);
		int col = atoi(argv[7]);
		int chn = atoi(argv[8]);
		int label_size = atoi(argv[9]);
		int num = atoi(argv[10]);
		data_rand_argu(argv[2], argv[3], argv[4], argv[5], row, col, chn, label_size, num);
	}
	return 0;
}
