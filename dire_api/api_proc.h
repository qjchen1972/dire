#pragma once
#ifndef _API_PROC__H
#define _API_PROC__H

#include "layer.h"
#include "active.h"
#include "batchnorm.h"
#include "conv.h"
#include "samp.h"
#include "end.h"
#include "scale.h"
#include "start.h"
#include "grating.h"


#define  MAX_NET  3000

#define  SAVE_PROC  0
#define  INIT_PROC  1
#define  FORWARD_PROC 2
#define  UPDATE_PROC  3
#define  RESET_SIZE   4
#define  CHECK_PROC   5
#define  BACKWARD_PROC 6
#define  INIT_SPACE    7
#define  INIT_BASE_SPACE    8
#define  INIT_INPUT_SPACE   9
#define  INIT_PARAM    10
#define  CLEAN_MEM  11



bool fw_is_ready_ok(Clayer *temp)
{
	for (int i = 0; i < temp->m_prev_num; i++)
		if (!temp->m_prev_layer[i]->is_proc_over) return false;
	return true;
}

bool bw_is_ready_ok(Clayer *temp)
{
	for (int i = 0; i < temp->m_next_num; i++)
		if (!temp->m_next_layer[i]->is_proc_over) return false;
	return true;
}

void init_travel(Clayer *start)
{	
	Clayer *p[MAX_NET];
	Clayer *temp;

	int num = 0;
	p[0] = start;
	while (num >= 0)
	{
		temp = p[num];
		num--;
		temp->is_proc_over = false;
		for (int i = 0; i < temp->m_next_num; i++)
		{
			if (!temp->m_next_layer[i]->is_proc_over) continue;
			num++;
			p[num] = temp->m_next_layer[i];
		}
	}
}

Clayer*  get_clayer(Clayer *start, int layernum)
{
	Clayer *p[MAX_NET];
	Clayer *temp;

	init_travel(start);

	int num = 0;
	p[0] = start;
	while (num >= 0)
	{
		temp = p[num];
		num--;
		if (temp->m_layer_num == layernum) return temp;
		temp->is_proc_over = true;
		for (int i = 0; i < temp->m_next_num; i++)
		{
			if (temp->m_next_layer[i]->is_proc_over) continue;
			num++;
			p[num] = temp->m_next_layer[i];
		}
	}
	return nullptr;
}

GTYPE forward_travel(Clayer *start, int type, ifstream *fin =nullptr, ofstream* fout=nullptr)
{
	GTYPE answer = 0;
	Clayer *p[MAX_NET];
	Clayer *temp;

	init_travel(start);

	int num = 0;
	p[0] = start;	

	while (num >= 0)
	{
		temp = p[num];
		num--;
		if (!fw_is_ready_ok(temp)) continue;
		temp->is_proc_over = true;
		switch (type)
		{
		case SAVE_PROC:
			temp->save_param(fout);
			break;
		case INIT_PROC:
			temp->init();
			break;
		case FORWARD_PROC:
			temp->forward_proc();
			if ( temp->m_net_status == PARAM_CHECK && !strcmp(temp->m_layer_type, END_LAYER))
				answer = dynamic_cast<End*>(temp)->get_cost();
			if (temp->m_net_status == TESTING && temp->fw_need_write_file)
			{
				if(fout)
				{
					GTYPE *output = new GTYPE[temp->m_fw_buff_size];
					temp->get_fw_output(output);
					(*fout).write((char*)output, sizeof(GTYPE)*temp->m_fw_buff_size);
					delete[] output;
				}
			}						
			break;
		case UPDATE_PROC:
			temp->update_weight();	
			break;
		case RESET_SIZE:
			temp->reset_size();
			break;
		case CHECK_PROC:
			temp->check_weight();
			break;
		case INIT_SPACE:
			temp->init_space();
			break;
		case INIT_PARAM:
			temp->init_param(fin);
			break;
		case CLEAN_MEM:
			temp->clean_mem();
			break;
		default:
			return answer;
		}
		for (int i = 0; i < temp->m_next_num; i++)
		{
			if (temp->m_next_layer[i]->is_proc_over) continue;
			num++;
			p[num] = temp->m_next_layer[i];
		}
	}
	return answer;
}


void backward_travel(Clayer *start, Clayer *end, int type)
{
	Clayer *p[MAX_NET];
	Clayer *temp;

	init_travel(start);

	int num = 0;
	p[0] = end;	

	while (num >= 0)
	{
		temp = p[num];
		num--;
		if (!bw_is_ready_ok(temp)) continue;
		temp->is_proc_over = true;
		switch (type)
		{
		case BACKWARD_PROC:
			//start层 不需要后向
			if (temp->m_layer_num > 0) temp->backword_proc();		
			break;
		default:
			return;
		}
		for (int i = 0; i < temp->m_prev_num; i++)
		{
			if (temp->m_prev_layer[i]->is_proc_over) continue;
			num++;
			p[num] = temp->m_prev_layer[i];
		}
	}
}


GTYPE forward_check(Clayer *start)
{
	forward_travel(start, RESET_SIZE);
	return forward_travel(start, FORWARD_PROC);
}

void create_check_travel(Clayer *start)
{
	Clayer *p[MAX_NET];
	Clayer *temp;

	int num = 0;
	p[0] = start;
	while (num >= 0)
	{
		temp = p[num];
		num--;
		temp->is_check_over = true;
		temp->create_test_weight(forward_check, start);
		for (int i = 0; i < temp->m_next_num; i++)
		{
			if (temp->m_next_layer[i]->is_check_over) continue;
			num++;
			p[num] = temp->m_next_layer[i];
		}
	}
}


#include "../test.h"

GTYPE test(Clayer *start, Clayer *end)
{
	start->set_train_num(1);

	char file[FILENAME_LEN];
	
	ifstream  fdata, flabel;

	//GTYPE *data = nullptr;
	//GTYPE *output = nullptr;


	char *data = nullptr;
	char *output = nullptr;
	int start_input_size = start->m_fw_buff_size * dynamic_cast<Start*>(start)->m_input_type;
	int end_input_size = end->m_fw_buff_size * dynamic_cast<End*>(end)->m_input_type;

	sprintf(file,"%s/%s", g_config.m_test_dir, TEST_DATA);
	fdata.open(file, ios::in | ios::binary);
	if (!fdata.is_open())
	{
		printf("open %s error\n", file);
		return 0;
	}

	sprintf(file, "%s/%s", g_config.m_test_dir, TEST_LABEL);
	flabel.open(file, ios::in | ios::binary);
	if (!flabel.is_open())
	{
		printf("open %s error\n", file);
	}
	else
	{
		//output = new GTYPE[end->m_fw_buff_size];
		output = new char[end_input_size];
	}

	ofstream  *p,fout;
	sprintf(file, "%s/%s", g_config.m_test_dir, TEST_OUTPUT);
	fout.open(file, ios::out | ios::binary | ios::trunc);
	if (!fout.is_open())
	{
		printf("open %s error\n", file);
		p = nullptr;
		return 0;
	}
	else
		p = &fout;
	
	data = new char[start_input_size];
	//data = new GTYPE[start->m_fw_buff_size];

	init_test(start, end);
	GTYPE temp = 0;
	while (1)
	{
		fdata.read((char*)data, start_input_size);
		if (fdata.eof()) break;
		if (output)
		{
			flabel.read((char*)output, end_input_size);
			dynamic_cast<End*>(end)->set_label(output, dynamic_cast<End*>(end)->m_input_type);
		}
		dynamic_cast<Start*>(start)->set_input(data, dynamic_cast<Start*>(start)->m_input_type);
		forward_travel(start, RESET_SIZE);
		forward_travel(start, FORWARD_PROC,nullptr,p);
		if(output)
			test_proc(start,end, dynamic_cast<End*>(end)->m_in);
		else
			test_proc(start, end, nullptr);
	}
	temp = show_test_answer(start, end);
	
	if(p) fout.close();
	fdata.close();
	if (output)
	{
		flabel.close();
		delete[] output;
	}
	delete[] data;
	return temp;
}


void train(Clayer *start, Clayer *end, Clayer *global)
{
	//训练
	int epoch = 0;
	char file[FILENAME_LEN];

	ifstream  fdata,flabel;

	sprintf(file, "%s/%s", g_config.m_train_dir, TRAIN_DATA);
	fdata.open(file, ios::in | ios::binary);
	if (!fdata.is_open())
	{
		printf("open %s error\n", file);
		return ;
	}

	sprintf(file, "%s/%s", g_config.m_train_dir, TRAIN_LABEL);
	flabel.open(file, ios::in | ios::binary);
	if (!flabel.is_open())
	{
		printf("open %s error\n", file);
		return;
	}
	//GTYPE *data = new GTYPE[start->m_fw_buff_size];
	//GTYPE *label = new GTYPE[end->m_fw_buff_size];
	
	time_t now;
	time_t over;
	
	int start_input_size = start->m_fw_buff_size * dynamic_cast<Start*>(start)->m_input_type;
	int end_input_size = end->m_fw_buff_size * dynamic_cast<End*>(end)->m_input_type;

	char *data = new char[start_input_size];
	char *label = new char[end_input_size];

	while ( 1 )
	{
		start->set_train_num(0);
		fdata.clear();
		flabel.clear();
		fdata.seekg(0, ios::beg);
		flabel.seekg(0, ios::beg);		
		now = time(nullptr);
		while (1)
		{
			/*fdata.read((char*)data, start->m_fw_buff_size * sizeof(GTYPE));
			if (fdata.eof()) break;
			flabel.read((char*)label, end->m_fw_buff_size * sizeof(GTYPE));*/

			fdata.read((char*)data, start_input_size);
			if (fdata.eof()) break;
			flabel.read((char*)label, end_input_size);

			start->m_train_num++;
			dynamic_cast<Start*>(start)->set_input(data, dynamic_cast<Start*>(start)->m_input_type);
			dynamic_cast<End*>(end)->set_label(label, dynamic_cast<End*>(end)->m_input_type);
			forward_travel(start, FORWARD_PROC);
			backward_travel(start, end, BACKWARD_PROC);
			forward_travel(start, UPDATE_PROC);
		}

		g_config.read_param();
		global->get_param();

		over = time(nullptr);
		start->m_total_train += start->m_train_num;
		ofstream f;
		f.open("answer.txt", ios::out | ios::app);
		f << endl;
		f << "epoch: " << epoch << " run time is " << over - now << endl;
		f << "num: " << epoch << "( "<< start->m_total_train <<" ) whole cost is " << dynamic_cast<End*>(end)->m_wholecost << endl;
		f << endl;
		f.close();
		
		if ( fpclassify(dynamic_cast<End*>(end)->m_wholecost) == FP_NAN || 
			fpclassify(dynamic_cast<End*>(end)->m_wholecost) == FP_INFINITE ||
			fpclassify(dynamic_cast<End*>(end)->m_wholecost) == FP_SUBNORMAL) return;
		
		//save the model of the best answer
		if (g_config.m_answer_mode == ANSWER_MODE_ON)
		{
			unsigned int batch_num = start->m_batch_num;
			start->m_batch_num = g_config.m_test_batch_num;
			start->set_net_status(TESTING);
			forward_travel(start, RESET_SIZE);
			GTYPE temp = test(start, end);
			if (end->m_best_answer > temp)
			{
				end->m_best_answer = temp;

				ofstream  fout;
				sprintf(file, "%s/%s", g_config.m_bestanswer_param_dir, PARAM_FILE);
				fout.open(file, ios::out | ios::binary | ios::trunc);

				if (!fout.is_open())
				{
					printf("open %s error\n", file);
				}
				else
				{
					forward_travel(start, SAVE_PROC, nullptr, &fout);
					fout.close();
				}
			}
			if(g_config.m_free_type != NERVER_FREE) forward_travel(start, CLEAN_MEM);
			start->m_batch_num = batch_num;
			start->set_net_status(CONT_TRAINING);
			forward_travel(start, RESET_SIZE);
		}

		// save the model of the min loss
		if (g_config.m_loss_mode == LOSS_MODE_ON)
		{
			if (end->m_mincost > dynamic_cast<End*>(end)->m_wholecost)
			{
				end->m_mincost = dynamic_cast<End*>(end)->m_wholecost;
				ofstream  fout;
				sprintf(file, "%s/%s", g_config.m_minloss_param_dir, PARAM_FILE);
				fout.open(file, ios::out | ios::binary | ios::trunc);

				if (!fout.is_open())
				{
					printf("open %s error\n", file);
				}
				else
				{
					forward_travel(start, SAVE_PROC, nullptr, &fout);
					fout.close();
				}
			}
		}
		
		// save the model of last model 
		if (g_config.m_last_mode == LAST_MODE_ON)
		{
			ofstream  fout;
			sprintf(file, "%s/%s", g_config.m_last_param_dir, PARAM_FILE);
			fout.open(file, ios::out | ios::binary | ios::trunc);

			if (!fout.is_open())
			{
				printf("open %s error\n", file);
			}
			else
			{
				forward_travel(start, SAVE_PROC, nullptr, &fout);
				fout.close();
			}
		}

		epoch++;
		global->update_param(g_config.m_work_param_dir);				
	}
}

void check_gd(Clayer *start, Clayer *end )
{	
	// create test
	start->set_train_num(1);

	ifstream  fdata, flabel;
	char file[FILENAME_LEN];

	sprintf(file, "%s/%s", g_config.m_train_dir, TRAIN_DATA);
	fdata.open(file, ios::in | ios::binary);
	if (!fdata.is_open())
	{
		printf("open %s error\n", file);
		return;
	}

	sprintf(file, "%s/%s", g_config.m_train_dir, TRAIN_LABEL);
	flabel.open(file, ios::in | ios::binary);
	if (!flabel.is_open())
	{
		printf("open %s error\n", file);
		return;
	}
	//GTYPE *data = new GTYPE[start->m_fw_buff_size];
	//GTYPE *label = new GTYPE[end->m_fw_buff_size];

	int start_input_size = start->m_fw_buff_size * dynamic_cast<Start*>(start)->m_input_type;
	int end_input_size = end->m_fw_buff_size * dynamic_cast<End*>(end)->m_input_type;

	char *data = new char[start_input_size];
	char *label = new char[end_input_size];

	fdata.read((char*)data, start_input_size);
	flabel.read((char*)label, end_input_size);
	fdata.close();
	flabel.close();

	dynamic_cast<Start*>(start)->set_input(data, dynamic_cast<Start*>(start)->m_input_type);
	dynamic_cast<End*>(end)->set_label(label, dynamic_cast<End*>(end)->m_input_type);

	create_check_travel(start);
	forward_travel(start, RESET_SIZE);
	forward_travel(start, FORWARD_PROC);
	backward_travel(start, end, BACKWARD_PROC);
	forward_travel(start, CHECK_PROC);
}

void add_net(Clayer *before, Clayer *after,int *layer_num = nullptr )
{
	if (before->m_next_num >= MAX_CONN_NUM)
	{
		printf("next layer max limit\n");
		return;
	}
	before->m_next_layer[before->m_next_num] = after;	

	if (after->m_prev_num >= MAX_CONN_NUM)
	{
		printf("prev layer max limit\n");
		return;
	}
	after->m_prev_layer[after->m_prev_num] = before;

	before->m_next_pos[before->m_next_num] = after->m_prev_num;
	
	before->m_next_num++;
	after->m_prev_num++;

	if (layer_num)
	{
		*layer_num = *layer_num + 1;
		after->set_layer_num(*layer_num);
	}
}

#endif
