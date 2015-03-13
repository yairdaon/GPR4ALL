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


	// allocate memory and initialize
	int i,j; 
	double *xi    = malloc(veclen * sizeof(double));
	double *grad  = calloc((size_t)veclen       , sizeof(double));

	// create covariance vectors
	double *k    = covVec(X, x , r, d, nvecs, veclen); // k(X,x)
	double *kxn   = covVec(X, xn, r, d, nvecs, veclen); // k(X,xn)

	// multiply by inverse of covariance matrix: K^{-1}(X,X)k(X,x)
	double *kKinv   = solver(U, S, V, k   , reg, nvecs);	
	double *kxnKinv  = solver(U, S, V, kxn , reg, nvecs);

	// calculate the value of g:
	double gValue = cov(x, xn, r, d, veclen);
	double sigSqr = d;
	for(i = 0 ; i < nvecs ; i++) {
		gValue -=  kKinv[i] * kxn[i];     // enumerator
		sigSqr -=  kxn[i]   * kxnKinv[i]; // denominator (variance at xn)
	}

	// final value of g
	gValue = gValue*gValue/sigSqr;

	// the left term in  the expression for grad g
	double tmp = -cov(xn ,x, r, d, veclen)/(r*r);
	for (j = 0; j < veclen ; j++) {
		grad[j] = tmp*(xn[j] - x[j]);
	}
  
	// the right sum in the expression for grad g	
	for(i = 0 ; i < nvecs ; i++) {
	

		// set xi:
		for(j = 0 ; j < veclen ; j++) {
			xi[j] = X[i*veclen + j];
		}
	
		// set the multiplier of the vector difference
		tmp = kKinv[i]*cov(xn, xi, r, d, veclen)/(r*r);

		// add the vector to the grad, coordinate by coordinate
		for(j = 0 ; j < veclen ; j++) {
			grad[j] += tmp*(xn[j] - xi[j]);
		} 

	}

	// build return value
	struct gStruct ret;
	ret.gValue = gValue;
	ret.grad   = grad;

	free(xi);
	free(kxn); 
	free(k);
	free(kKinv);
	free(kxnKinv);
	return ret;

}




double * gradKOverSig( double *U, double *S, double *V,
			 		double *X, double *x, double *xn,
			 		double r, double d, double reg,
				 	int nvecs, int veclen) {



	int i, j;


 	// calcualte the first term first
	double sigSqr      = justVar(U, S, V, X, xn, r, d, reg, nvecs, veclen);
	double firstFactor = -cov(xn,x, r ,d ,veclen)/(r*r*sigSqr);
	double *firstTerm  =  calloc((size_t)veclen       , sizeof(double));

	for (i = 0 ; i < veclen ; i++ ) {
		firstTerm[i] = firstFactor*(xn[i] - x[i]);
	} 



	// now the second term calculation
	double secondFactor  =  -2.0*cov(xn,x, r, d, veclen)/(r*r*sigSqr*sigSqr);
	double *k            =  covVec(X, xn, r, d, nvecs, veclen); // k(X,xn)
	double *kKinv        =  solver(U, S, V, k , reg, nvecs);   	// K^{-1}k(X,xn)
	double *secondTerm   =  calloc((size_t)veclen       , sizeof(double));
	double *xi           =  calloc((size_t)veclen       , sizeof(double));
	double tmp, covXi;


	// loop over all vectors
	for ( i = 0 ; i < nvecs ; i ++ ) {


		//  first, pull xi from the big array X
		for ( j = 0 ; j < veclen ; j++) {
			xi[j] = X[i*veclen + j];
		}	

		
		// then calculate its covariance with xn
		covXi  = cov(xi,xn, r, d, veclen);

		// multiply
		tmp    = secondFactor*kKinv[i]*covXi;

	
		// set every coordinatethe second term according to the badass calculation
		for ( j = 0 ; j < veclen ; j++ ) {
			secondTerm[j] += tmp*(xn[j] - xi[j]); 
		}
	

	}





	// allocate memory for the result
	double *result           =  calloc((size_t)veclen       , sizeof(double));	
	
	// set the result to the correct values
	for( i = 0 ; i < veclen ; i++){
		result[i] = firstTerm[i] + secondTerm[i];
	}



	// free willy!!!!
	free(k);
	free(firstTerm);
	free(secondTerm);
	free(kKinv);
	free(xi);
	return result;
}
