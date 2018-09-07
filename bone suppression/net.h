#pragma once
#ifndef _NET__H
#define _NET__H

#include "dire_api/layer.h"
#include "dire_api/active.h"
#include "dire_api/batchnorm.h"
#include "dire_api/conv.h"
#include "dire_api/samp.h"
#include "dire_api/end.h"
#include "dire_api/scale.h"
#include "dire_api/start.h"
#include "dire_api/grating.h"
#include "dire_api/api_proc.h"
#include "dire_api/move.h"
#include "dire_api/limit.h"
#include "dire_api/linearmapping.h"
#define  IMG_NODE 32

//ȥ��
void train_net(Clayer **s, Clayer **e)
{
	// ��ʼ��������	
	Start *start = new Start();
	start->set_layer_num(0);
	start->set_node_num(2);
	start->set_fw_out(32, 32);
	//start->set_data_augm(true);
	*s = start;

	End *end = new End();
	end->set_fw_out(8, 8);
	end->set_node_num(2);
	end->set_cost_type(MSE);
	*e = end;

	Conv *conv;
	Batchnorm *bn;
	Scale *scale;
	Active *active;
	Grating *grating;
	

	//��ת��¼
	Clayer *jump;

	int layer_num = start->m_layer_num;

	//һ��ģ�� 2��conv (row  node) 32
	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(start, conv, &layer_num);

	bn = new Batchnorm();
	add_net(conv, bn, &layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv, &layer_num);

	bn = new Batchnorm();
	add_net(conv, bn, &layer_num);

	scale = new Scale();
	add_net(bn,scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	jump = conv;


	//һ��ģ�� 2��conv (row  node) 28
	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv, &layer_num);

	bn = new Batchnorm();
	add_net(conv, bn,&layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv, &layer_num);


	bn = new Batchnorm();
	add_net(conv, bn,&layer_num);
	add_net(jump, bn);
	jump = conv;

	scale = new Scale();
	add_net(bn,scale,&layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	//end

	//һ��ģ�� 2��conv   24
	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv, &layer_num);

	bn = new Batchnorm();
	add_net(conv, bn,&layer_num);

	scale = new Scale();
	add_net(bn, scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv, &layer_num);


	bn = new Batchnorm();
	add_net(conv, bn,&layer_num);
	add_net(jump, bn);
	jump = conv;

	scale = new Scale();
	add_net(bn,scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	//end

	//һ��ģ�� 2��conv 20
	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv, &layer_num);

	bn = new Batchnorm();
	add_net(conv,bn, &layer_num);

	scale = new Scale();
	add_net(bn,scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv,&layer_num);


	bn = new Batchnorm();
	add_net(conv, bn,&layer_num);
	add_net(jump, bn);
	jump = conv;


	scale = new Scale();
	add_net(bn,scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	//end	

	//һ��ģ�� 2��conv  16
	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active,conv, &layer_num);

	bn = new Batchnorm();
	add_net(conv, bn,&layer_num);

	scale = new Scale();
	add_net(bn,scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active, conv,&layer_num);

	bn = new Batchnorm();
	add_net(conv,bn, &layer_num);
	add_net(jump, bn);
	jump = conv;

	scale = new Scale();
	add_net(bn,scale, &layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale,active, &layer_num);

	//end
	//һ��ģ�� 2��conv  12
	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active,conv, &layer_num);

	bn = new Batchnorm();
	add_net(conv, bn, &layer_num);

	scale = new Scale();
	add_net(bn, scale,&layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale,active, &layer_num);

	conv = new Conv();
	conv->set_kernel_param(3, 3, 1, 1);
	conv->set_fw_out(start->m_fw_row, start->m_fw_col);
	conv->set_node_num(IMG_NODE);
	add_net(active,conv, &layer_num);


	bn = new Batchnorm();
	add_net(conv, bn, &layer_num);
	add_net(jump, bn);


	scale = new Scale();
	add_net(bn, scale,&layer_num);

	active = new Active();
	active->set_atcive_type(RELU);
	add_net(scale, active, &layer_num);

	//end


	//���м������
	grating = new Grating();
	grating->set_type(CENTRE_TYPE);
	grating->set_fw_out(start->m_fw_row - 24, start->m_fw_col - 24);
	add_net(active,grating, &layer_num);

	conv = new Conv();
	conv->set_kernel_param(1, 1, 1, 1);
	conv->set_fw_out(start->m_fw_row - 24, start->m_fw_col - 24);
	conv->set_node_num(2);
	add_net(grating,conv, &layer_num);

	add_net(conv, end, &layer_num);
}

void build_net(Clayer **start, Clayer **end)
{
	train_net(start, end);
}
#endif
