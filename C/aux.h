

double cov(double *x, double *y, double r, double d, int N);

double *covVec(double *X, double *x, double r, double d, int nvecs, int veclen);

double *covMat(double *X, double r, double d, int nvecs, int veclen);

double *solver(double *U, double *S, double *V, double *b , double reg, int n);

