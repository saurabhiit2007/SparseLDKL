//////////////////////////////////////////////////////////////////////////////
// This implementation of SparseLDKL was written by Saurabh Goyal.          //
////////////////////////////////////////////////////////////////////////////// 
#include<stdio.h>
#include<vector>
#include<math.h>
#include<string.h>
#include"../include/Utils.h"

void PathWeight(INSTANCE X, MyFloat* pathwt,const MyFloat *Theta, const MyFloat rbfgamma, const int L, const int D) {
  for(int i = 0 ; i < L ; i++)
	  pathwt[i] = 1.0;
  int M=(L+1)/2;
  double tanh_dist;
  const MyFloat *ttheta;
  for(int i = 0 ; i < M-1 ; i++){
	  ttheta = Theta + i*D;
	  tanh_dist = tanh(rbfgamma*_dot(X,(MyFloat*)ttheta ,D));
	  pathwt[2*i+1] = pathwt[i]*(MyFloat)(1.0+tanh_dist)/2;
	  pathwt[2*i+2] = pathwt[i]*(MyFloat)(1.0-tanh_dist)/2;
  }
}

void GradTheta(INSTANCE X, double *grad, const MyFloat* pathwt, const MyFloat* localwt, const MyFloat *W,const MyFloat *Theta,const int M, const int D,const MyFloat rbfgamma) {
  memset(grad,0,sizeof(double)*(M-1));
  int L=2*M-1;
  MyFloat *tanh_thetaTx = new MyFloat[M-1];
  for(int m=0;m<M-1;m++) {
    tanh_thetaTx[m] = tanh(rbfgamma*_dot(X,(MyFloat*)Theta+m*D,D));
  }
  for(int l=0;l<L;l++) {
    int r=l;
	double temp_grad = pathwt[l] * localwt[l] * _dot(X,(MyFloat*)W+l*D,D);
    while(r>0) {
      int t=(r-1)/2;
      grad[t] +=  temp_grad *(r%2==1 ? (1-tanh_thetaTx[t]) : (-1-tanh_thetaTx[t]));
      r=t;
    }
  }
 delete [] tanh_thetaTx;
}


/*double ComputePrimalObj(const double* x, const double* y, const double *W, const double *ThetaPrime, const double *Theta, const double lW, const double lThetaPrime, const double lTheta, const double Sigma, const int M, const int D, const int N,const double rbfgamma) {
  int L = 2*M-1;
  double obj = 0.5*lW*dnrm2_(W, L*D) + 0.5*lThetaPrime*dnrm2_(ThetaPrime, L*D) + 0.5*lTheta*dnrm2_(Theta, (M-1)*D);
  int losscount=0.0;
  double total_loss = 0.0;
  double* pathwt_k = new double[L];
  double *localwt_k = new double[L];
  for(int n=0;n<N;n++) {
    PathWeight(x+n*D,pathwt_k,Theta, rbfgamma, L, D);
    for(int l=0;l<L;l++) {
      localwt_k[l] = tanh(Sigma*ddot_(ThetaPrime+l*D,x+n*D,D));
    }
    double y_pred=0.0;
    for(int l=0;l<L;l++) 
    {
      y_pred += pathwt_k[l] * localwt_k[l] * ddot_(W+l*D,x+n*D,D);
    }
    double loss = 1.0-y[n]*y_pred;
    if (loss>0.0) {
      total_loss += loss;
    }				
    if (loss>1) {
      losscount++;
    }
  }
  total_loss /= (double)N;
  delete [] pathwt_k;
  delete [] localwt_k;
  return (obj+total_loss);
}*/

void Train(DATA X, LABEL Y, MyFloat *W, MyFloat *Theta, MyFloat *ThetaPrime, const int M, const MyFloat lW, const MyFloat lTheta, const MyFloat lThetaPrime,const MyFloat Sigma,const int MaxIter,const int N, const int D, int retrain, int *WCounter, int *ThpCounter, int *ThCounter){
  int t=1;
  int L = 2*M-1;
  MyFloat rbfgamma = (MyFloat)0.01;
  MyFloat *w_tplushalf = new MyFloat[L*D];	
  MyFloat *thetaprime_tplushalf = new MyFloat[L*D];	
  MyFloat *theta_tplushalf = new MyFloat[(M-1)*D];	
  memcpy(w_tplushalf,W,L*D*sizeof(MyFloat));
  memcpy(thetaprime_tplushalf,ThetaPrime,L*D*sizeof(MyFloat));
  memcpy(theta_tplushalf,Theta,(M-1)*D*sizeof(MyFloat));
  std::vector<int> all_indices;
  for(int n=0;n<N;n++) {
    all_indices.push_back(n);
  }
  int K1 = (int)sqrt((float)N);		//points per iteration
  for(int iter=1;iter<=MaxIter;iter++) {
	  if(iter%100==1) {
      if(M==1) {
        rbfgamma=1.0;
      }
      else {
        MyFloat sum_tr=0.0;
        for(int f=0;f<100;f++) {
          int theta_i = rand()%(M-1);
          int x_i = rand()%N;
          sum_tr += abs(_dot(X[x_i],Theta+theta_i*D,D));
        }
        sum_tr/=100.0;
        rbfgamma = (MyFloat)0.1/sum_tr;
        rbfgamma *= (MyFloat)pow(2.0,iter/(MaxIter/10));
      }
    }
    t++;
	
	// Randomnly Sample Training Points for Stochastic Gradient Optimization
	
	std::vector<int> At_indices;
    int k=0;
    while(k<K1) {
      int r = k + floor((rand()/double(RAND_MAX))*(N-k-1));
      std::swap(all_indices[k], all_indices[r]);
      At_indices.push_back(all_indices[k]);
      k++;
    }

	// Update Step Size
	
    MyFloat eta_t_w=(MyFloat)1.0/(lW*(MyFloat)pow(t,0.5));
    MyFloat eta_t_theta=(MyFloat)1.0/(lTheta*(MyFloat)pow(t,0.5));
    MyFloat eta_t_thetaprime=(MyFloat)1.0/(lThetaPrime*(MyFloat)pow(t,0.5));
	MyFloat coeff=(MyFloat)(t - 1)/t;
    
	// Scale parameters
	
	_scale(thetaprime_tplushalf,coeff,L*D);
	_scale(w_tplushalf,coeff,L*D);
	_scale(theta_tplushalf,coeff,(M-1)*D);

	// Memory Allocation for Local Variables

    MyFloat* localwt_k = new MyFloat[L];
    MyFloat* pathwt_k = new MyFloat[L];
    double* grad_theta = new double[(M-1)];
    MyFloat* wt_dot_xi = new MyFloat[L];
    
	// Main Training Loop
    
	for (int k=0;k<K1;k++) {
      int index=At_indices[k];
	  
	  // Compute path weight
	  
	  PathWeight(X[index], pathwt_k, Theta, rbfgamma, L, D);
	  
	  // Compute local weight
	  
	  for(int l=0;l<L;l++) {
        localwt_k[l] = tanh(Sigma*_dot(X[index],ThetaPrime+l*D,D));
      }
	  
	  // Get prediction
	  
	  MyFloat y_pred=0.0;
      for(int l=0;l<L;l++) {
		wt_dot_xi[l] = _dot(X[index],W+l*D,D);
        y_pred += pathwt_k[l] * localwt_k[l] * wt_dot_xi[l];
      }
	  
	  MyFloat loss=(MyFloat)1.0-Y[index]*y_pred;
      
	  // Update if loss > 0
      
	  if(loss>0) {
        // Compute gradient w.r.t Theta
		
		GradTheta(X[index], grad_theta, pathwt_k, localwt_k, W, Theta, M, D, rbfgamma);
		
		// Update parameters
		
		for (int l=0;l<L;l++) {
			MyFloat temp_w_tplushalf = Y[index]*eta_t_w/K1*pathwt_k[l]*localwt_k[l];
			_add(X[index], (w_tplushalf+l*D), D, temp_w_tplushalf);
        }
        for (int l=0;l<L;l++) {
			MyFloat local_grad = (1-localwt_k[l]*localwt_k[l])*Y[index]*eta_t_thetaprime/K1*pathwt_k[l]*wt_dot_xi[l];
			_add(X[index], (thetaprime_tplushalf+l*D), D, local_grad);
		}
        for (int m=0;m<M-1;m++) {
			MyFloat temp_theta_tplushalf = Y[index]*eta_t_theta/K1*(MyFloat)grad_theta[m];	
			_add(X[index], (theta_tplushalf+m*D),  D, temp_theta_tplushalf);
		}
		
		if(retrain == 1){
			for(int w1=0;w1<L*D;w1++){
				if(WCounter[w1] == 0) w_tplushalf[w1] = 0.0;
				if(ThpCounter[w1] == 0) thetaprime_tplushalf[w1] = 0.0;
			}
			for(int w1=0;w1<(M-1)*D;w1++)
				if(ThCounter[w1] == 0) theta_tplushalf[w1] = 0.0;
		}
      }
    }
    delete [] grad_theta;
    delete [] pathwt_k;
    delete [] localwt_k;
    delete [] wt_dot_xi;
    
	// Copy parameters
	
	memcpy(W,w_tplushalf,L*D*sizeof(MyFloat)); 
    memcpy(ThetaPrime,thetaprime_tplushalf,L*D*sizeof(MyFloat)); 
    memcpy(Theta,theta_tplushalf,(M-1)*D*sizeof(MyFloat));
  }
  delete[] w_tplushalf;
  delete[] theta_tplushalf;	
  delete[] thetaprime_tplushalf;
}
