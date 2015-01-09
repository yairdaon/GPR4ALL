struct krigStruct{
   double   krig;
   double   sigSqr;
   double * gradKrig;
   double * gradSig2;
} ;

struct krigStruct krigGrads(double *U, double *S, double *V,
			 double *X, double *x, double *y,
			 double *grad , double prior,
			 double r, double d, double reg,
			 int nvecs, int veclen);

double krig(double *U, double *S, double *V,
			 double *X, double *x, double *y,
			 double prior, double r, double d, double reg,
			 int nvecs, int veclen);

double *krigVar(double *U, double *S, double *V,
			 double *X, double *x, double *y,
			 double prior, double r, double d, double reg,
			 int nvecs, int veclen);
