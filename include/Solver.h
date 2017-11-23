//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
////////////////////////////////////////////////////////////////////////////// 
#ifndef _SOLVER_H_
#define _SOLVER_H_
void PathWeight(INSTANCE X, MyFloat* pathwt,const MyFloat *Theta, const MyFloat rbfgamma, const int L, const int D);
void GradTheta(INSTANCE X, MyFloat *grad, const MyFloat* pathwt, const MyFloat* localwt, const MyFloat *W,const MyFloat *Theta,const int M, const int D,const MyFloat rbfgamma);
MyFloat ComputePrimalObj(DATA X, LABEL y, const MyFloat *W, const MyFloat *ThetaPrime, const MyFloat *Theta, const MyFloat lW, const MyFloat lThetaPrime, const MyFloat lTheta, const MyFloat Sigma, const int M, const int D, const int N,const MyFloat rbfgamma);
void Train(DATA Data, LABEL Y, MyFloat *W, MyFloat *Theta, MyFloat *ThetaPrime, const int M, const MyFloat lW, const MyFloat lTheta, const MyFloat lThetaPrime,const MyFloat Sigma,const int MaxIter,const int N, const int D, int retrain, int *WCounter, int *ThpCounter, int *ThCounter);
#endif
