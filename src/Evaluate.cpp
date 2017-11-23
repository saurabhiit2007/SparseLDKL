//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
////////////////////////////////////////////////////////////////////////////// 
#include<math.h>
#include "../include/Utils.h"
#include "../include/Model.h"
int Evaluate(struct Model &model){
	DATA X = model.X;
	LABEL Y = model.Y;
	MyFloat *Score = model.Score;
	MyFloat *W = model.W;
	MyFloat *ThetaPrime = model.ThetaPrime;
	MyFloat *Theta = model.Theta;
	MyFloat Sigma = model.Sigma;
	int M = model.M;
	int D = model.D;
	int N = model.N;
	int Correct = 0;
	for(int n = 0 ; n < N; n++){
		int curr = 0;
		Score[n] = 0;
		while(curr < M-1){
			Score[n] += tanh(Sigma*_dot(X[n],ThetaPrime+curr*D,D)) * _dot(X[n],W+curr*D,D);
			MyFloat t = _dot(X[n],Theta+curr*D,D);
			curr = (t>0) ? 2*curr+1 : 2*curr+2;
		}
		Score[n] += tanh(Sigma*_dot(X[n],ThetaPrime+curr*D,D)) * _dot(X[n],W+curr*D,D);
		if(Y[n]*Score[n]>0)
			Correct++;
	}
	return Correct;
}
