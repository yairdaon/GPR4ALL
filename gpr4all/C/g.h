struct valGrad{
   double   value;
   double * grad;
} ;

struct valGrad avgVar(double *U, double *S, double *V,
		      double *X, double * sample, double *xn,
		      double r, double d, double reg,
		      int nvecs, int veclen, int nsteps);

struct valGrad g(double *U, double *S, double *V,
			 double *X, double *x, double *xn,
			 double r, double d, double reg,
			 int nvecs, int veclen);
