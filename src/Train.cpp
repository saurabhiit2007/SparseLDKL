//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
////////////////////////////////////////////////////////////////////////////// 
#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<math.h>
#include <time.h>
#include<fstream>
#include "string.h"
#include "algorithm"
#include"../include/Utils.h"
//#include"../include/Timer.h"
#include"../include/Solver.h"
#include"../include/Model.h"

using namespace std;

void ParseInput(int argc, char** argv, struct Model &prob);

void ChipParameters(struct Model &model, int *WCounter, int *ThpCounter, int *ThCounter){
	
	MyFloat *tempW = new MyFloat[(model.L)*(model.D-1)];
	MyFloat *tempTP = new MyFloat[(model.L)*(model.D-1)];
	MyFloat *tempT = new MyFloat[(model.M-1)*(model.D-1)];
	int count = 0;
	for(int w = 0 ; w < model.L*model.D ; w++){
		if((w+1)%model.D != 0){
			tempW[count] = fabs(model.W[w]);
			tempTP[count] = fabs(model.ThetaPrime[w]);
			count++;
		}
	}
	sort(tempW, tempW+model.L*(model.D-1));
	sort(tempTP, tempTP+model.L*(model.D-1));
	
	count = 0;
	for(int w = 0 ; w < (model.M-1)*model.D ; w++){
		if((w+1)%model.D != 0){
			tempT[count] = fabs(model.Theta[w]);
			count++;
		}
	}
	sort(tempT, tempT+(model.M-1)*(model.D-1));
	
	int totalNonZero_wo_bias = model.totalNonZero - model.L*2 - model.M + 1;
	int W_ThP_nonZero = totalNonZero_wo_bias * (float)model.L*2.0/(float)(model.L*2.0 + model.M - 1);
	int Th_nonZero = totalNonZero_wo_bias - W_ThP_nonZero;
	
	for(int w = 0 ; w < model.L*model.D ; w++){
		if(fabs(model.W[w]) < tempW[(model.L)*(model.D-1) - W_ThP_nonZero/2] && (w+1)%model.D != 0){
			model.W[w] = 0.0;
			WCounter[w] = 0;
		}else WCounter[w] = 1;
		if(fabs(model.ThetaPrime[w]) < tempTP[(model.L)*(model.D-1) - W_ThP_nonZero/2] && (w+1)%model.D != 0){
			model.ThetaPrime[w] = 0.0;
			ThpCounter[w] = 0;
		}else ThpCounter[w] = 1;
	}

	for(int w = 0 ; w < (model.M-1)*model.D ; w++){
		if(fabs(model.Theta[w]) < tempT[(model.M-1)*(model.D-1) - Th_nonZero] && (w+1)%model.D != 0){
			model.Theta[w] = 0.0;
			ThCounter[w] = 0;
		}else ThCounter[w] = 1;
	}
}

void exit_with_help(){
	cout << "Train [Options] TrainingDataFile [ModelFile]" << endl << endl;
	cout << "Options:"<<endl;
	cout << "-D   : Depth of the SparseLDKL tree."<<endl;
	cout << "-lW  : \\lambdaW = regularizer for classifier parameter W  (try : 0.1, 0.01 or 0.001)."<<endl;
	cout << "-lT  : \\lambdatheta = regularizer for kernel parameter \theta  (try : 0.1, 0.01 or 0.001)."<<endl;
	cout << "-lP  : \\lambdatheta' = regularizer for kernel parameters \theta'  (try : 0.1, 0.01 or 0.001)."<<endl;
	cout << "-S   : \\sigma = parameter for sigmoid sharpness  (try : 1.0, 0.1 or 0.01)."<<endl;
    	cout << "-Z   : \\totalNonZero = total number of non-zero entries in final pruned model (depends on the Memory Budget constraint)."<<endl;
	cout << "-N   : [Optional: default 1] Number of times to train SparseLDKL with different random initializations." << endl;
	cout << "-I   : [Optional: default 15000] Number of passes through the dataset."<<endl;
	cout << "TrainingDataFile : File containing training data."<<endl;
	cout << "ModelFile        : [Optional: default TrainingDataFile.model] Files containing the models learnt by SparseLDKL. For example, given the command Train -N 5 ... banana.train, SparseLDKL will learn 5 different models and store them in banana.train.model1, ..., banana.train.model5."<<endl;
	exit(1);
}

int main(int argc, char** argv){
	Model model;
	if(WINDOWS)
		srand(unsigned(1));
	else
		srand(time(NULL));
	//Timer Tm;
	ParseInput(argc,argv,model);
	char buf[1024]; 
	cout << "Running SparseLDKL with following parameters on " << model.DataFile << endl;
	cout << "D : " << log2(unsigned(model.M)) << endl;
	cout << "Sigma : " << model.Sigma << endl;
	cout << "lW : " << model.lW << endl;
	cout << "lThetaPrime : " << model.lThetaPrime << endl;
	cout << "lTheta : " << model.lTheta << endl;
	cout << "NRun : " << model.NRun << endl;
    	cout << "Z (Non-zero) : " << model.totalNonZero << endl;
	cout << "MaxIter : " << model.MaxIter << endl;
	cout << "DataFile : " << model.DataFile << endl;
	cout << "ModelFile : " << model.ModelFile << endl;
	cout << "Loading Data"<< endl;
	ReadData(model.DataFile, model.X, model.Y, model.D, model.N);
	cout << "Data Loading done. " << endl;
	
	model.L = 2*model.M-1;
	int *WCounter = new int[model.L*model.D];
	int *ThpCounter = new int[model.L*model.D];
	int *ThCounter = new int[(model.M-1)*model.D];
	int retrain = 0;
	
    	clock_t stime, etime;
    
	for(int i = 0 ; i < model.NRun ; i++){
		cout << "Run : " << i+1 << endl;
		
		InitModel(model);

		// Prune mask for W, Thetaprime and Theta
		for(int w1=0;w1<model.L*model.D;w1++){
			WCounter[w1] = 1;
			ThpCounter[w1] = 1;
		}
		for(int w1=0;w1<(model.M-1)*model.D;w1++)
		    ThCounter[w1] = 1;

		//Tm.StartTimer();
	        stime = clock();
        
		// Iterative Hard Thresholding with relaxed budget
		Train(model.X, model.Y, model.W, model.Theta, model.ThetaPrime, model.M, model.lW, model.lTheta, model.lThetaPrime, model.Sigma, model.MaxIter, model.N, model.D, retrain, WCounter, ThpCounter, ThCounter);
		
		// Prune parameters to fit the memory budget
		ChipParameters(model, WCounter, ThpCounter, ThCounter);

		model.MaxIter = 5000;

		// Gradient Step with fixed budget
		retrain = 1;
		Train(model.X, model.Y, model.W, model.Theta, model.ThetaPrime, model.M, model.lW, model.lTheta, model.lThetaPrime, model.Sigma, model.MaxIter, model.N, model.D, retrain, WCounter, ThpCounter, ThCounter);

		// Iterative Hard Thresholding with relaxed budget
		retrain = 0;
		Train(model.X, model.Y, model.W, model.Theta, model.ThetaPrime, model.M, model.lW, model.lTheta, model.lThetaPrime, model.Sigma, model.MaxIter, model.N, model.D, retrain, WCounter, ThpCounter, ThCounter);

		// Prune parameters to fit the memory budget
		ChipParameters(model, WCounter, ThpCounter, ThCounter);

		// Final Gradient Step with fixed budget
		retrain = 1;
		Train(model.X, model.Y, model.W, model.Theta, model.ThetaPrime, model.M, model.lW, model.lTheta, model.lThetaPrime, model.Sigma, model.MaxIter, model.N, model.D, retrain, WCounter, ThpCounter, ThCounter);

        	etime = clock();
        	float elapsedTime = (etime-stime)*1000.0/CLOCKS_PER_SEC;
        	cout << "Training Time : " << elapsedTime << endl;
		//Tm.StopTimer();
			
		model.TrainTime[i] = elapsedTime;
		//cout << "Training Time : "<< model.TrainTime[i] << endl;
		sprintf(buf,"%s%d",model.ModelFile,i+1);
		SaveModel(buf, model);
		delete[] model.W;
		delete[] model.ThetaPrime;
		delete[] model.Theta;
	}
	cout << "Average Training Time : " << avg(model.TrainTime,model.NRun) << endl;

	for(int i = 0 ; i < model.N ; i++)
		delete[] model.X[i];

	delete[] model.X;
	delete [] model.Y;
	delete [] model.TrainTime;
	return 1;
}

void ParseInput(int argc, char** argv,struct Model &prob){
	if(argc < 12)
		exit_with_help();
	int i;
	prob.Sigma = 0;
	prob.M = 0;
	prob.NRun = 1;
	prob.lTheta = 0;
	prob.lThetaPrime = 0;
	prob.lW = 0;
	prob.MaxIter = 15000;
	for( i = 1 ; i < argc ; i++){
		if(argv[i][0] != '-')
			break;
		if(++i >= argc)
			exit_with_help();
		switch(argv[i-1][1]){
			case 'D':
				prob.M = int(pow(2.0,atoi(argv[i])));
				break;
			case 'S':
				prob.Sigma = (MyFloat)atof(argv[i]);
				break;
			case 'N':
				prob.NRun = atoi(argv[i]);
				break;
			case 'I':
				prob.MaxIter = atoi(argv[i]);
				break;
			case 'Z':
				prob.totalNonZero = atoi(argv[i]);
				break;	
			case 'l':
				switch(argv[i-1][2]){
					case 'T':
						prob.lTheta = (MyFloat)atof(argv[i]);
						break;
					case 'W':
						prob.lW = (MyFloat)atof(argv[i]);
						break;
					case 'P':
						prob.lThetaPrime = (MyFloat)atof(argv[i]);
						break;
					default:
						cout << "Unknown option: -%c\n"<<argv[i-1][2]<<endl;
						exit_with_help();
						break;
				}
				break;
			default :
				cout << "Unknown option: "<<argv[i-1][1]<<endl;
				exit_with_help();
				break;
		}
	}
	if( i >= argc)
		exit_with_help();
	strcpy(prob.DataFile,argv[i]);
	if(i+1 < argc)
		sprintf(prob.ModelFile,"%s.model",argv[i+1]);
	else{
		char* p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(prob.ModelFile,"%s.model",p);
	}
	prob.TrainTime = new double[prob.NRun];
}
