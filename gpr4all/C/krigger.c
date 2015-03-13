#include "krigger.h"
#include "aux.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>



struct krigStruct krigGrads(double *U, double *S, double *V,
			 double *X, double *x, double *y,
			 double *grad , double prior,
			 double r, double d, double reg,
			 int nvecs, int veclen) {
	/*
	 * do krigging
	 * dimensions of variables:
	 * U	 is an nvecs    by   nvecs     matrix
	 * V	 is an nvecs    by   nvecz     matrix
	 * S	 is an nvecs    by   1         vector
	 * X	 is an nvecs    by   veclen    matrix
	 * x	 is an veclen   by   1         vector
	 * y	 is an nvecs    by   1         vector
	 * grad  is an veclen   by   1         vector
	 *
	 */


	// allocate memory
	double *xMinusX   = calloc((size_t)nvecs*veclen , sizeof(double));
	double *gradKrig  = calloc((size_t)veclen       , sizeof(double));
	double *gradSig2  = calloc((size_t)veclen       , sizeof(double));
	int i,j;
	double tmp, xmX;

	// create covariance vector
	double *k     = covVec(X, x, r, d, nvecs, veclen);


	// multiply by inverse of covariance matrix
	double *kKinv = solver(U, S, V, k , reg, nvecs);


	// the kriged value is...
	double krig = prior;
	for(i = 0 ; i < nvecs ; i++ ) {
		krig += y[i]*kKinv[i];
	}


	// get variance ...
	double sigSqr = d;
	for(i = 0 ; i < nvecs ; i++ ) {
		sigSqr = sigSqr - k[i]*kKinv[i];
	}
	// ...ensure variance is positive
	if (sigSqr < 0.0 ) {
		sigSqr = 0.0;
	}


	// set x - X
	for(i = 0 ; i < nvecs ; i++ ) {
		for(j = 0 ; j < veclen ; j++ ) {
			xMinusX[i*veclen + j] = x[j] - X[i*veclen + j];
		}
	}

	// set (F - prior) * Kinv
	double * FmpKinv = solver(U, S, V, y, reg, nvecs);


	// set the gradient of kriged value and variance
	for(i = 0 ; i < veclen ; i++ ) {
		gradKrig[i] +=  grad[i]; // the prior
		for(j = 0 ; j < nvecs ; j++ ) {
			tmp = k[j];
			xmX = xMinusX[j*veclen + i];
			gradKrig[i] +=    -( FmpKinv[j] * tmp * xmX )/( r*r );
			gradSig2[i] += 2.0*( kKinv[j]   * tmp * xmX )/( r*r );
		}
	}



	// build return value
	struct krigStruct ret;
	ret.krig       = krig;
	ret.sigSqr     = sigSqr;
	ret.gradKrig   = gradKrig;
	ret.gradSig2   = gradSig2;




	// free memory we don't need
	free(xMinusX);
	free(FmpKinv);
	free(k);
	free(kKinv);

	return ret;

}





double *krigVar(double *U, double *S, double *V,
			 double *X, double *x, double *y,
			 double prior, double r, double d, double reg,
			 int nvecs, int veclen) {
	/*
	 * do krigging
	 * dimensions of variables:
	 * U	 is an nvecs    by   nvecs     matrix
	 * V	 is an nvecs    by   nvecz     matrix
	 * S	 is an nvecs    by   1         vector
	 * X	 is an nvecs    by   veclen    matrix
	 * x	 is an veclen   by   1         vector
	 * y	 is an nvecs    by   1         vector
	 */


	// allocate memory and initialize
	int i;

	// create covariance vector
	double *k     = covVec(X, x, r, d, nvecs, veclen);

	// multiply by inverse of covariance matrix
	double *kKinv = solver(U, S, V, k , reg, nvecs);


	// the kriged value is y * K^{-1} * k
	double krig = 0;
	for(i = 0 ; i < nvecs ; i++ ) {
		krig += y[i]*kKinv[i];
	}


	// variance is  cov(0,0) - k * K^{-1} * k
	double sigSqr = d;
	for(i = 0 ; i < nvecs ; i++ ) {
		sigSqr = sigSqr - k[i]*kKinv[i];
	}

	// ...ensure variance is positive
	if (sigSqr < 0.0 ) {
		sigSqr = 0.0;
	}

	double *result = calloc((size_t) 2 , sizeof(double));
	result[0] = krig + prior;
	result[1] = sigSqr;

	free(k);
	free(kKinv);
	return result;

}

double krig(double *U, double *S, double *V,
			 double *X, double *x, double *y,
			 double prior, double r, double d, double reg,
			 int nvecs, int veclen) {
	/*
	 * do krigging
	 * dimensions of variables:
	 * U	 is an nvecs    by   nvecs     matrix
	 * V	 is an nvecs    by   nvecz     matrix
	 * S	 is an nvecs    by   1         vector
	 * X	 is an nvecs    by   veclen    matrix
	 * x	 is an veclen   by   1         vector
	 * y	 is an nvecs    by   1         vector
	 *
	 */

	// create covariance vector
	double *k     = covVec(X, x, r, d, nvecs, veclen);

	// multiply by inverse of covariance matrix
	double *kKinv = solver(U, S, V, k , reg, nvecs);

	// the kriged value is...
	int i;
	double krig = 0;
	for(i = 0 ; i < nvecs ; i++ ) {
		krig += y[i]*kKinv[i];
	}

	free(k);
	free(kKinv);
	return krig + prior;

}



double justVar(double *U, double *S, double *V,
			 double *X, double *x,
			 double r, double d, double reg,
			 int nvecs, int veclen) {
	/*
	 * do krigging, return only the variance
	 * dimensions of variables:
	 * U	 is an nvecs    by   nvecs     matrix
	 * V	 is an nvecs    by   nvecz     matrix
	 * S	 is an nvecs    by   1         vector
	 * X	 is an nvecs    by   veclen    matrix
	 * x	 is an veclen   by   1         vector
	 * y	 is an nvecs    by   1         vector
	 */


	// allocate memory and initialize
	int i;

	// create covariance vector
	double *k     = covVec(X, x, r, d, nvecs, veclen);

	// multiply by inverse of covariance matrix
	double *kKinv = solver(U, S, V, k , reg, nvecs);

	// variance is  cov(0,0) - k * K^{-1} * k
	double sigSqr = d;
	for(i = 0 ; i < nvecs ; i++ ) {
		sigSqr = sigSqr - k[i]*kKinv[i];
	}

	// ...ensure variance is positive
	if (sigSqr < 0.0 ) {
		sigSqr = 0.0;
	}


	free(k);
	free(kKinv);
	return sigSqr;

}

