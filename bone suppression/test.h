#pragma once
#ifndef _TEST__H
#define _TEST__H


static void bone_init_test(Clayer *start, Clayer *end)
{
	end->m_prev_layer[0]->fw_need_write_file = true;
}

static GTYPE  bone_test(Clayer *start, Clayer *end, GTYPE *label)
{
	return  0;
}

static GTYPE bone_show_test_answer(Clayer *start, Clayer *end)
{
	return  0;
}



void  init_test(Clayer *start, Clayer *end)
{
	bonf_init_test(start, end);
}

GTYPE  test_proc(Clayer *start, Clayer *end, GTYPE *label = nullptr)
{
	return bone_test(start, end, label);
}

GTYPE show_test_answer(Clayer *start, Clayer *end)
{
	return bone_show_test_answer(start, end);
}
#endif
