//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
////////////////////////////////////////////////////////////////////////////// 
#include<stdio.h>
#include<iostream>
#include<math.h>
#include <time.h>
#include<vector>
#include<stdlib.h>
#include"string.h"
#include"../include/Utils.h"
//#include"../include/Timer.h"
#include"../include/Model.h"
#include"../include/Evaluate.h"
#define LABEL(X) X > 0 ? 1 : -1

void exit_with_help(void){
	cout << "Usage: Predict ModelFile TestDataFile [options]" << endl << endl;
	cout << "ModelFile : The base name of the files containing the learnt SparseLDKL models. This argument should be the same as the ModelFile argument used during training. For example, if the learnt SparseLDKL models were stored in banana.train.model1, ..., banana.train.model5 then ModelFile should be set to banana.train."<<endl<<endl;
	cout << "TestDataFile : File containing test data."<<endl;
	cout << "-N : [Optional: default 1] Number of models over which SparseLDKL's predictions will be averaged. Note that this cannot be greater than the argument used in -N during training."<< endl;
	cout << "-R : [Optional: default Result.txt] Result filename." << endl;
	exit(1);
}

void ParseInput(int argc, char** argv, struct InputData &ip){
	if(argc < 3)
		exit_with_help();
	int i;
	ip.NModel = 1;
	sprintf(ip.ModelFile,"%s.model", argv[1]);
	sprintf(ip.OutputFile,"%s.output", argv[1]); 
	strcpy(ip.DataFile,argv[2]);
	strcpy(ip.ResultFile,"Result.txt");
	for(i = 3 ; i < argc ; i++){
		if(argv[i][0] != '-'){
			cout << "Unknown Option : " << argv[i] << endl;
			exit_with_help();
		}
		if(++i >= argc)
			exit_with_help();
		switch(argv[i-1][1]){
			case 'N':
				ip.NModel = atoi(argv[i]);
				break;
			case 'R':
				strcpy(ip.ResultFile, argv[i]);
				break;
			default:
				cout << "Unknown Option : " << argv[i-1][1] << endl;
				exit_with_help();
		}
	}
	ip.TestAcc = new MyFloat[ip.NModel];
	ip.TestTime = new double[ip.NModel];
}


int main(int argc, char** argv){
	struct InputData Input;
	char buf[1024];
	//Timer Tm;
	ParseInput(argc,argv,Input);
	struct Model model;
	int Correct;
	
	ReadData(Input.DataFile,model.X,model.Y,model.D,model.N);
	model.Score = new MyFloat[model.N];
	
	FILE* Fp = fopen(Input.ResultFile,"w");
	FILE *fout;
	fprintf(Fp,"Run\tAccuracy\tTime\n");
	clock_t stime, etime;

	for(int i = 0; i < Input.NModel ; i++){
		
        	cout << "Testing with Model " << i+1 << endl;
		sprintf(buf,"%s%d",Input.ModelFile,i+1);
		LoadModel(buf,model);
		
        	// Prediction
		//Tm.StartTimer();
		stime = clock();
		Correct = Evaluate(model);
		etime = clock();

		float elapsedTime = (etime-stime)*1000.0/CLOCKS_PER_SEC;
		//Tm.StopTimer();
        
		Input.TestAcc[i] = (MyFloat)Correct/(MyFloat)model.N;
		Input.TestTime[i] = elapsedTime;
		cout <<"Accuracy : " << Input.TestAcc[i]*100.0 << endl;
		cout <<"Prediction Time : " << Input.TestTime[i] << endl;
		fprintf(Fp,"%d\t%f\t%f\n",i+1,Input.TestAcc[i],Input.TestTime[i]);

		// Write predicted labels to file
 
		sprintf(buf,"%s%d",Input.OutputFile,i+1);
		fout = fopen(buf,"w");
		for(int i = 0 ; i < model.N ; i++)
			fprintf(fout, "%f %d\n",model.Score[i], LABEL(model.Score[i]));
		fclose(fout);

		delete [] model.W;
		delete [] model.Theta;
		delete [] model.ThetaPrime;
	}
	cout << "Average Test Accuracy :" <<avg(Input.TestAcc,Input.NModel)*100 << endl;
	cout << "Average Test Time (Milliseconds) :" <<avg(Input.TestTime,Input.NModel) << endl;
	fprintf(Fp,"Average Test Accuracy : %f\n",avg(Input.TestAcc,Input.NModel)*100); 
	fprintf(Fp,"Average Test Time (Milliseconds): %f\n" ,avg(Input.TestTime,Input.NModel));
	fclose(Fp);

	for(int i = 0 ; i < model.N ; i++)
		delete[] model.X[i];
    
	delete [] model.Y;
	delete [] model.Score;
	delete [] Input.TestAcc;
	delete [] Input.TestTime;
	return 0;
}
