struct gStruct{
   double   gValue;
   double * grad;
} ;

struct gStruct g(double *U, double *S, double *V,
			 double *X, double *x, double *xn,
			 double r, double d, double reg,
			 int nvecs, int veclen);
