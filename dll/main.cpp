#include<stdio.h>
#include<time.h>
#include "dire.h"
#include <Windows.h>

typedef int(*Api)(int, char*);
int main(int argc, char** argv)
{
	HINSTANCE hDllInst = LoadLibrary("dire.dll");
	Api myfun = 0;
	myfun = (Api)GetProcAddress(hDllInst, "dire");
	myfun(2, "./config.txt");
	//dire(2, "config.txt");
}