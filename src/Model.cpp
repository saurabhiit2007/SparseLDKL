//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
////////////////////////////////////////////////////////////////////////////// 
#include<stdio.h>
#include<iostream>
#include<fstream>
#include<sstream>
#include<iterator>
#include<math.h>
#include<string.h>
#include<stdlib.h>
#include"../include/Utils.h"
#include"../include/Model.h"

using namespace std;
void LoadModel(char* filename, struct Model &model){
	std::ifstream fip(filename);
	if(!fip.is_open()){
		cout<< "Erorr in Opening file: "<<filename << endl;
		exit(0);
	}
	std::istringstream in;
	std::string line;
	
	// Get Dimension and number of kernels
	
	std::getline(fip, line);
	in.str(line);
	in >> model.D;
	in >> model.M;
	model.L = 2* model.M - 1;
	in.clear();
	
	// Get kernel regularization parameters
	
	std::getline(fip, line);
	in.str(line);
	in >> model.lW;
	in >> model.lThetaPrime;
	in >> model.lTheta;
	in >> model.Sigma;
	in.clear();
	
	// Allocate memory for classifier parameters
	
	int D = model.D;
	int M = model.M;
	int L = model.L;
	model.W = new MyFloat[L*D];
	model.ThetaPrime = new MyFloat[L*D];
	model.Theta = new MyFloat[(M-1)*D];
	
	// Read classifier parameters
	
	for(int count = 0;(count < L) && std::getline(fip, line) ; count++ ){
		in.str(line);
		for(int j = 0 ; j < D ; j++)
			in >> model.W[D*count + j];
		in.clear();
	}
	for(int count = 0;(count < L) && std::getline(fip, line) ; count++ ){
		in.str(line);
		for(int j = 0 ; j < D ; j++)
			in >> model.ThetaPrime[D*count + j];
		in.clear();
	}
	for(int count = 0;(count < M-1) && std::getline(fip, line) ; count++ ){
		in.str(line);
		for(int j = 0 ; j < D ; j++)
			in >> model.Theta[D*count + j];
		in.clear();
	}
	model.Score = new MyFloat[model.N];
	fip.close();
}

void DisplayModel(struct Model model){
	int L = model.L;
	int D = model.D;
	int M = model.M;
	cout << "Model Parameters." << endl;
	cout<<"M: "<< model.M << endl <<"L : "<< model.L << endl << " D: "<< model.D<<endl;
	cout<<"lw: " << model.lW << endl <<  "lThetaPrime: " << model.lThetaPrime <<endl<< "lTheta: " << model.lTheta <<endl<< "Sigma: " << model.Sigma<<endl;
	cout << endl<<"W Matrix"<<endl<<endl;
	for(int i = 0 ; i < L ; i++){
		for(int j = 0 ; j < D ;j++)
			cout << model.W[i*D+j]<< " ";
		cout << endl;
	}
	cout << endl<<"ThetaPrime Matrix"<<endl<<endl;
	for(int i = 0 ; i < L ; i++){
		for(int j = 0 ; j < D ;j++)
			cout << model.ThetaPrime[i*D+j]<< " ";
		cout << endl;
	}
	cout << endl <<"Theta Matrix"<<endl<<endl;
	for(int i = 0 ; i < M-1 ; i++){
		for(int j = 0 ; j < D ;j++)
			cout << model.Theta[i*D+j]<< " ";
		cout << endl;
	}
}

void SaveModel(char *filename, struct Model model){
	ofstream fout(filename);
	if(!fout.is_open()){
		cout << "Error in opening file: "<<filename << endl;
		exit(0);
	}
	fout <<model.D << " " << model.M << endl;
	fout << model.lW << " " << model.lThetaPrime << " " << model.lTheta << " " <<model.Sigma << endl;
	int L = model.L;
	int D = model.D;
	int M = model.M;
	for(int i = 0 ; i < L ; i++){
		for(int j = 0 ; j < D-1 ; j++)
			fout <<model.W[i*D+j] << " ";;
		fout <<model.W[i*D+D-1]<< endl;;
	}
	for(int i = 0 ; i < L ; i++){
		for(int j = 0 ; j < D-1 ; j++)
			fout <<model.ThetaPrime[i*D+j] << " ";;
		fout <<model.ThetaPrime[i*D+D-1]<< endl;;
	}
	for(int i = 0 ; i < M - 1 ; i++){
		for(int j = 0 ; j < D-1 ; j++)
			fout <<model.Theta[i*D+j] << " ";;
		fout <<model.Theta[i*D+D-1]<< endl;;
	}
	fout.close();
} 

void InitModel(struct Model &model){
	int D = model.D;
	int M = model.M;
	int L = 2*M-1;
	model.L = L; 
	model.W = new MyFloat[L*D];
	model.Theta = new MyFloat[(M-1)*D];	
	model.ThetaPrime = new MyFloat[L*D];	
	int temp;
	for(int i = 0 ; i < L*D ; i++){
		temp = rand();
		model.W[i] = MyFloat((temp%2? 1:-1)*MyFloat(temp)/MyFloat(RAND_MAX));
		temp = rand();
		model.ThetaPrime[i] = MyFloat((temp%2? 1:-1)*MyFloat(temp)/MyFloat(RAND_MAX));
	}
	for(int i = 0 ; i < (M-1)*D ; i++){
		temp = rand();
		model.Theta[i] = MyFloat((temp%2? 1:-1)*MyFloat(temp)/MyFloat(RAND_MAX));
	}
}
