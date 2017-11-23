//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
////////////////////////////////////////////////////////////////////////////// 
#ifndef _MODEL_H_
#define _MODEL_H_
struct Model{
	int M,L,D,N,NRun;
	int MaxIter;
	MyFloat lW,lTheta,lThetaPrime,Sigma;
	DATA X;
	LABEL Y;
	MyFloat *W,*ThetaPrime,*Theta;
	double *TrainTime;
	MyFloat *Score;
	char DataFile[1024];
	char ModelFile[1024];
	int totalNonZero;
};

void LoadModel(char* filename, struct Model &model);
void DisplayModel(struct Model model);
void SaveModel(char *filename, struct Model model);
void InitModel(struct Model &model);
#endif
