//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
////////////////////////////////////////////////////////////////////////////// 
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "string.h"
#include <math.h>
#include <vector>
#include"../include/Utils.h"

using namespace std;

//Log2
unsigned int log2( unsigned int x )
{
  unsigned int ans = 0 ;
  while( x>>=1 ) ans++;
  return ans ;
}

// Compute L2 norm of Vector

MyFloat dnrm2_(const MyFloat *v, int n) {
  MyFloat ntemp = 0.0;
  for(int i=0; i<n; i++) 
    ntemp += pow(v[i], 2);
  return ntemp;
} 

// Average of numbers

MyFloat avg(MyFloat *Data, int N){
	MyFloat sum = 0;
	for(int i = 0 ; i < N ; i++)
		sum += Data[i];
	return(sum/N);
}
double avg(double *Data, int N){
	double sum = 0;
	for(int i = 0 ; i < N ; i++)
		sum += Data[i];
	return(sum/N);
}

// Load Data

void exit_with_readError(int num, char* filename){
	cout << "Error in Reading Line " << num << " in file " << filename << endl;
	exit(1);
}

void ReadData(char* FileName, DATA &X, MyFloat* &y,int &D, int &N){
    FILE* Fp;
    char* tok;
    char Line[100000];
    int Ncount = 0;
    int Dcount = 0;
    Fp = fopen(FileName,"r");
    if(Fp == NULL){
        cout << "Error in Opening file: " << FileName << endl;
        exit(0);
    }
    if(fgets(Line,100000,Fp) != NULL){
        tok = strtok(Line," ");
        N = atoi(tok);
        tok = strtok(NULL," ");
        D = atoi(tok)+1;
    }
    X = new INSTANCE[N];
    y = new MyFloat[N];
    while(fgets(Line,100000,Fp) != NULL){
        X [Ncount] = new MyFloat[D];
        tok = strtok(Line," ");
        y[Ncount] = (MyFloat)atof(tok);
        tok = strtok(NULL," ");
        while(tok != NULL){
            X[Ncount][Dcount] = (MyFloat)atof(tok);
            Dcount++;
            tok = strtok(NULL," ");
        }
        X[Ncount][Dcount] = 1.0;
        Dcount++;
        if(Dcount != D){
            cout << "Dimension Mismatch while reading data from file: "<< FileName << " at line number: " << Ncount+2<<endl;
            exit(1);
        }
        Ncount++;
        Dcount = 0;
    }
    if(Ncount != N){
        cout<< "Mismatch in total number of datapoints in file: "<< FileName<<endl;
        exit(2);
    }
    fclose(Fp);
}

