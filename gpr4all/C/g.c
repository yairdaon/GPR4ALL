#include "krigger.h"
#include "aux.h"
#include "g.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

struct gStruct g(double *U, double *S, double *V,
			 		double *X, double *x, double *xn,
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
	double tmp, gValue;
	double cv             = cov(x, xn, r, d, veclen);
	double *y             = covVec( X, x, r, d, nvecs, veclen);

	// empty \ zero arrays
	double *dummyZeroGrad = calloc( (size_t)veclen , sizeof(double));//all zeros
	double *grad          = malloc(veclen * sizeof(double));
	double *xi            = malloc(veclen * sizeof(double));

	// we do "kriging" at xn but the "function values" are the vector y
    // which is a covariance vector. This is seen to work by checking the 
	// derivation in the pdf. An advantage of this approach is that we don't
	// need to rewrite a program to calculate gradients.
	struct krigStruct all = krigGrads( U, S, V, X, xn, y, dummyZeroGrad ,
										0.0, r, d, reg, nvecs, veclen);
		
	// unpack
	double krig        = all.krig;
	double sigSqr      = all.sigSqr; 	
	double * gradKrig  = all.gradKrig;
	double * gradSig2  = all.gradSig2;

	// value of g
	tmp = cv - krig;
	gValue = tmp*tmp/sigSqr;

	
	// calcualte the coefficients of the gradients, from pdf
	double firstCoef   = -2.0*cv/(r*r*sigSqr);
	double secondCoef  = -2.0/sigSqr;
	double thirdCoef   = -tmp*tmp/(sigSqr*sigSqr);	

	for ( j = 0 ; j < veclen ; j++) {
		grad[j] = firstCoef*(xn[j] - x[j]) + secondCoef*gradKrig[j] 
						+ thirdCoef*gradSig2[j];
	}



	free(y);
	free(xi);
	free(dummyZeroGrad);
	free(gradKrig);
	free(gradSig2);


	// build return value
	struct gStruct ret;
	ret.gValue = gValue;
	ret.grad   = grad;

	return ret;
	
}
