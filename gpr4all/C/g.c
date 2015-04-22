#include "krigger.h"
#include "aux.h"
#include "g.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>



struct valGrad avgVar(double *U, double *S, double *V,
		      double *X, double * sample, double *xn,
		      double r, double d, double reg,
		      int nvecs, int veclen, int nsteps) {
  /*
   * return the average varaince and its gradient
   * dimensions of variables:
   * U	 is an nvecs    by   nvecs     matrix
   * V	 is an nvecs    by   nvecz     matrix
   * S	 is an nvecs    by   1         vector
   * X	 is an nvecs    by   veclen    matrix
   * x	 is a  veclen   by   1         vector
   * xn	 is a  veclen   by   1         vector
   * sample a  nsteps   by   veclen    matrix
   */
  
  // allocate memory and get datum here. only last bit is interesting
  int i,j; 
  double objective = 0.0;
  double expNegG, gradcoef;
  struct valGrad tmp;

  // empty \ zero arrays
  double *grad = calloc( (size_t)veclen , sizeof(double));//all zeros
  double *Zi   = malloc( veclen * sizeof(double));

  for (i = 0; i < nsteps ; i++) {
   
    // copy the current vector into Zi
    memcpy(Zi , sample + i*veclen, veclen*sizeof(double));
    for( j = 0 ; j < veclen ; j++){
      if  (Zi[j] != sample[i*veclen + j])
	      printf("Error!\n");
    }
    
    // get g and its grad for current sample Zi
    tmp = g( U, S, V, X, Zi, xn, r, d, reg, nvecs, veclen);
    
    expNegG = exp(-tmp.value); 

    // add to the sum
    objective += expNegG;
    
    // the ith coeffiicient:
    gradcoef  = expNegG;

    // add the gradient
    for( j = 0 ; j < veclen ; j++) {
      grad[j] -= gradcoef*tmp.grad[j];
    }

    // free the output of function g
    free(tmp.grad);

  }

  // free whatever we don't return
  free(Zi);


  // build return value
  struct valGrad ret;
  ret.value  = objective;
  ret.grad   = grad;

  return ret;

}
	


  


struct valGrad g(double *U, double *S, double *V,
		 double *X, double *Z, double *xn,
		 double r, double d, double reg,
		 int nvecs, int veclen) {
  /*
   * return g and its gradient
   * dimensions of variables:
   * U	 is an nvecs    by   nvecs     matrix
   * V	 is an nvecs    by   nvecz     matrix
   * S	 is an nvecs    by   1         vector
   * X	 is an nvecs    by   veclen    matrix
   * x	 is a  veclen   by   1         vector
   * xn	 is a  veclen   by   1         vector
   */


  // allocate memory and get datum here. only last bit is interesting
  int j; 
  double gValue, u;
  double cv             = cov(Z, xn, r, d, veclen); // k(Zj, xn)
  double *y             = covVec( X, Z, r, d, nvecs, veclen); // k^(n-1)(Zj)


  // empty \ zero arrays
  //double *dummyZeroGrad = calloc( (size_t)veclen , sizeof(double));//all zeros
  double *grad          = malloc(        veclen * sizeof(double));
  double *dummyZeroGrad = calloc((size_t)veclen , sizeof(double));

  // we do "kriging" at xn but the "function values" are the vector y
  // which is a covariance vector. This is seen to work by checking the 
  // derivation in the pdf. An advantage of this approach is that we don't
  // need to rewrite a program to calculate gradients.
  struct krigStruct all = krigGrads( U, S, V, X, xn, y, dummyZeroGrad ,
  				     0.0, r, d, reg, nvecs, veclen);
  //struct krigStruct all = locKrig( U, S, V, X, xn, y,
  //				   r, d, reg, nvecs, veclen);


  // unpack
  double krig        = all.krig;
  double sigSqr      = all.sigSqr; 	
  double * gradKrig  = all.gradKrig;
  double * gradSig2  = all.gradSig2;

  // value of g
  u = cv- krig;
  gValue   = u*u/sigSqr;
  
  // calcualte the coefficients of the gradients, see pdf file
  double firstCoef   = -2.0*u*cv/(r*r*sigSqr);
  double secondCoef  = -2.0*u/sigSqr;
  double thirdCoef   = -u*u/(sigSqr*sigSqr);	

  for ( j = 0 ; j < veclen ; j++) {
    grad[j] = firstCoef*(xn[j] - Z[j]) + secondCoef*gradKrig[j] 
      + thirdCoef*gradSig2[j];
  }



  free(y);
  free(dummyZeroGrad);
  free(gradKrig);
  free(gradSig2);


  // build return value
  struct valGrad ret;
  ret.value  = gValue;
  ret.grad   = grad;

  return ret;
	
}
