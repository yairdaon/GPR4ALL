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
	 * x	 is an veclen   by   1         vector
	 * xn	 is an veclen   by   1         vector
	 */


	// allocate memory and initialize
	int i,j; 
	double *xi    = malloc(veclen * sizeof(double));
	double *grad  = calloc((size_t)veclen       , sizeof(double));

	// create covariance vector
	double *k     = covVec(X, x, r, d, nvecs, veclen);

	// multiply by inverse of covariance matrix
	double *kKinv = solver(U, S, V, k , reg, nvecs);


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
	ret.gValue = 0.0;
	ret.grad   = grad;

	free(xi);
	free(k);
	free(kKinv);
	return ret;

}
