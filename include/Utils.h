//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
//////////////////////////////////////////////////////////////////////////////
#ifndef _UTILS_H_
#define _UTILS_H_
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <xmmintrin.h>
#define MyFloat float
#define WINDOWS 0 //Change it to 0 for linux system^M

using namespace std;


typedef MyFloat Node;
typedef Node* INSTANCE;
typedef INSTANCE* DATA;

typedef MyFloat* LABEL;

// Compute base 2 logirithm
unsigned int log2( unsigned int x );

// Compute Dot Product of Vectors
inline MyFloat _dot(INSTANCE X, MyFloat* V, int count) {
    MyFloat result = 0;
    for(int i = 0 ; i < count ; i++)
        result += X[i]*V[i];
    return result;
}


// Implements v = v + scale * X
inline void _add(INSTANCE X, MyFloat* V, int count, MyFloat scale){
    for(int i = 0 ; i < count ; i++){
        V[i] += X[i]*scale;
    }
}


// Scale a vector by a scaler
inline void _scale(MyFloat* V, const MyFloat scale, int count){
	for(int i = 0 ; i < count ; i++)
		V[i] *= scale; 
}


// Compute L2 norm of Vector
MyFloat dnrm2_(const MyFloat *v, int n);


// Multiply Vector by a Scaler
void dscal_(MyFloat *v, const MyFloat s,  int n);


// Average of Numbers
MyFloat avg(MyFloat *Data, int N);
double avg(double *Data, int N);


// Load Dataset
void ReadData(char* FileName, DATA &X, LABEL &y,int &D, int &N);

#endif
