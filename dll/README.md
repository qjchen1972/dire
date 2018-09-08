
it is a sample of windows dll using GPU
====

* build dll
  * x86:  nvcc -m32 -arch=sm_61 -o dire.dll --shared dll.cu
  * X64:  nvcc -m64 -arch=sm_61 -o dire.dll --shared dll.cu
  
* other program  call it for test:

typedef int( *Api)(int, char *);

int main(int argc, char** argv)

{

	HINSTANCE hDllInst = LoadLibrary("dire.dll");
	
	Api myfun = 0;
	
	myfun = (Api)GetProcAddress(hDllInst, "dire");
	
	myfun(2, "./config.txt");
	
}
