//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
////////////////////////////////////////////////////////////////////////////// 
#ifndef _EVALUATE_H_
#define _EVALUATE_H_
struct InputData{
	char DataFile[1024];
	char ModelFile[1024];
	char ResultFile[1024];
	char OutputFile[1024];
	int NModel;
	MyFloat *TestAcc;
	double *TestTime;
};
int Evaluate(struct Model &model);
#endif
