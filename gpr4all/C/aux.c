#include "aux.h"
#include <math.h>
#include <stdlib.h>

double *solver(double *U, double *S, double *V, double *b , double reg, int n) {
	/*
	 * solve Ax = b using tychonoff regularization
	 * A = U * S * V (no transpose here!)
	 * U is an n by n matrices
	 * V is an n by n matrices
	 * S is an n by 1 vector
	 * b is an n by 1 vector
	 * reg is the regularization we use
	 */

	// allocate memory and initialize
	double *c = calloc((size_t)n , sizeof(double)); // also
	double *x = calloc((size_t)n , sizeof(double)); // sets
	double *y = calloc((size_t)n , sizeof(double)); // zeros
	int i,j;
	double tmp;

	//c = U^t * b
	for (i = 0 ; i < n ; i++) {
		tmp = b[i];
		for (j = 0 ; j < n ; j++ ) {
			c[j] += U[ i*n + j ]*tmp;
		}
	}

	// y_i  = s_i * c_i  /  (s_i^2 + reg
	for (i = 0 ; i < n ; i++) {
		tmp =S[i];
		y[i] = c[i]*tmp/( tmp*tmp + reg );
	}

	// x = V^t * y
	for (i = 0 ; i < n ; i++) {
			tmp = y[i];
			for (j = 0 ; j < n ; j++ ) {
				x[j] += V[ i*n + j ]*tmp;
			}
		}

	free(y);
	free(c);
	return x;

}


double *covMat(double *X, double r, double d, int nvecs, int veclen) {

	double result;
	double dist;
	int n;

	double *K;
	K = malloc(nvecs * nvecs * sizeof(double));

	int row, col;
	for( row = 0 ; row < nvecs ; row++) {
		for (col = 0 ; col < row+1 ; col++) {

			result = 0.0;
			for (n = 0; n < veclen; n++) {
					dist = X[col*veclen + n] - X[row*veclen + n];
					result += dist * dist;
			}

			result = d*exp(  -result/(2.0*r*r)  );

			// set the element of the cov matrix
			K[row*nvecs + col] = result;
			K[col*nvecs + row] = result;
		}
	}
	return K;
}

double *covVec(double *X, double *x, double r, double d, int nvecs, int veclen) {

	double result;
	double dist;
	int n;

	double *k;
	k = malloc(nvecs * sizeof(double));

	int row;
	for( row = 0 ; row < nvecs ; row++) {

		result = 0.0;
		for (n = 0; n < veclen; n++) {
				dist = x[n] - X[row*veclen + n];
				result += dist * dist;
		}

		result = d*exp(  -result/(2.0*r*r)  );
		k[row] = result;
	}
	return k;
}


double cov(double *x, double *y,  double r, double d, int N) {

	double result = 0.0;
	double dist;
	int n;
    	for (n = 0; n < N; n++) {
     	   	dist = y[n] - x[n];
		result += dist * dist;
    	}

	result = result /(2.0*r*r);
	result = d*exp(-result);
    	return result;
}
