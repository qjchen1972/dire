#include<stdio.h>
#include<time.h>
#define  GPU
#include "../Bacilli detection/net.h"
#include "dire.h"

//nvcc -m32 -arch=sm_61 -o dire.dll --shared dll.cu

int dire(int mode, char *file) 
{
    time_t now = time(0);
    srand(now);

	g_config.set_config_file(file);

	if (!g_config.read_param()) return 0;
	
	Clayer *global = new Clayer();
	global->set_batch_size(g_config.m_batch_num);

	switch (mode)
	{
	case 0:
		global->set_net_status(INIT_TRAINING);
		global->m_total_train = 0;
		global->m_mincost = 423864;
		global->m_best_answer = 42356;
		global->update_param(g_config.m_work_param_dir);
		break;
	case 1:
		global->set_net_status(CONT_TRAINING);
		break;
	case 2:
		global->set_net_status(TESTING);
		break;
	case 3:
		global->set_net_status(PARAM_CHECK);
		global->set_batch_size(2);
		break;
	default:
		return 0;
	}
	
	// ��ʼ��������	
	Start *start;
	End *end;

	build_net((Clayer**)&start, (Clayer**)&end);

	forward_travel(start, INIT_PROC);	
	if (mode == CONT_TRAINING || mode == TESTING)
	{
		char file[FILENAME_LEN];
		ifstream  fin;
		sprintf(file, "%s/%s", g_config.m_work_param_dir, PARAM_FILE);
		fin.open(file, ios::in | ios::binary);
		if (!fin.is_open())
		{
			printf("open %s error\n", file);
			return 0;
		}
		forward_travel(start, INIT_PARAM, &fin,nullptr);
		fin.close();
	}
	else
		forward_travel(start, INIT_PARAM);	
	
	//ָ����gpu�豸���������
	forward_travel(start, INIT_SPACE);
    global->set_global_mem();
    init_layer_const();

	switch (mode)
	{
	case 0:		
	case 1:
		train(start, end, global);
		break;
	case 2:
	   {
		test(start, end);
		break;
		}
	case 3:
		check_gd(start, end);
		break;
	default:
		return 0;
	}
	return 1;
}