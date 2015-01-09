#include <Python.h>
#include <numpy/arrayobject.h>
#include "aux.h"
#include <stdio.h>

static char module_docstring[] =
    "This module provides an interface for calculating covariance using C.";

static char cov_docstring[] =
    "Calculate the covariance between two vectors.";

static char cov_vec_docstring[] =
	"Calculate the covariances between a vector and a list of vectors.";

static char cov_mat_docstring[] =
		"get a covariance matrix.";

static char solver_docstring[] =
				"solve using tychonoff regularization.";

static PyObject *_aux_cov(PyObject *self, PyObject *args);

static PyObject *_aux_covVec(PyObject *self, PyObject *args);

static PyObject *_aux_covMat(PyObject *self, PyObject *args);

static PyObject *_aux_solver(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
	    {"cov"    , _aux_cov   , METH_VARARGS, cov_docstring    },
	    {"cov_vec", _aux_covVec, METH_VARARGS, cov_vec_docstring},
	    {"cov_mat", _aux_covMat, METH_VARARGS, cov_mat_docstring},
	    {"solver" , _aux_solver, METH_VARARGS, solver_docstring },
	    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_aux(void) {

    PyObject *m = Py_InitModule3("_aux", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}


static PyObject *_aux_solver(PyObject *self, PyObject *args)
{
    double reg;
	PyObject *U_obj, *S_obj , *V_obj , *b_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOd", &U_obj, &S_obj, &V_obj, &b_obj, &reg ))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *U_array = PyArray_FROM_OTF(U_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *V_array = PyArray_FROM_OTF(V_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *b_array = PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    /* If that didn't work, throw an exception. */
    if (U_array == NULL || S_array == NULL || V_array == NULL || b_array == NULL ) {
        Py_XDECREF(U_array);
        Py_XDECREF(S_array);
        Py_XDECREF(V_array);
        Py_XDECREF(b_array);
        return NULL;
    }

    /* What are the dimensions? */
    int n      = (int)PyArray_DIM(U_array, 0);
    int m      = (int)PyArray_DIM(V_array, 1);
    int p      = (int)PyArray_DIM(S_array, 0);
    int l      = (int)PyArray_DIM(b_array, 0);


    if ( (n!=m) || (m!=p) || (p!=l) || (l!=n)   ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimensions don't match!!");
    		return NULL;
        }
    /* Get pointers to the data as C-types. */
    double *U    = (double*)PyArray_DATA(U_array);
    double *S    = (double*)PyArray_DATA(S_array);
    double *V    = (double*)PyArray_DATA(V_array);
    double *b    = (double*)PyArray_DATA(b_array);


    /* Call the external C function to compute the covariance. */
    double *x = solver(U, S, V, b, reg, n);

    /* Clean up. */
    Py_DECREF(U_array);
    Py_DECREF(S_array);
    Py_DECREF(V_array);
    Py_DECREF(b_array);

	npy_intp dims[1] = {n};

	PyObject *ret = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	memcpy(PyArray_DATA(ret), x, n*sizeof(double));
	free(x);

	return ret;
}







static PyObject *_aux_covMat(PyObject *self, PyObject *args)
{
    double r, d;
    PyObject *X_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Odd", &X_obj, &r, &d ))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    /* If that didn't work, throw an exception. */
    if (X_array == NULL ) {
        Py_XDECREF(X_array);
        return NULL;
    }

    /* What are the dimensions? */
    int nvecs  = (int)PyArray_DIM(X_array, 0);
    int veclen = (int)PyArray_DIM(X_array, 1);

    /* Get pointers to the data as C-types. */
    double *X    = (double*)PyArray_DATA(X_array);


    /* Call the external C function to compute the covariance. */
    double *K = covMat(X, r,d, nvecs, veclen);

    /* Clean up. */
    Py_DECREF(X_array);

	npy_intp dims[2] = {nvecs ,nvecs};

	PyObject *ret = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	memcpy(PyArray_DATA(ret), K, nvecs*nvecs*sizeof(double));
	free(K);

	return ret;
}



static PyObject *_aux_covVec(PyObject *self, PyObject *args)
{
    double r,d;
	PyObject *X_obj, *x_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOdd", &X_obj, &x_obj, &r, &d ))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    /* If that didn't work, throw an exception. */
    if (X_array == NULL || x_array == NULL ) {
        Py_XDECREF(X_array);
        Py_XDECREF(x_array);
        return NULL;
    }

    /* What are the dimensions? */
    int nvecs  = (int)PyArray_DIM(X_array, 0);
    int veclen = (int)PyArray_DIM(X_array, 1);
    int xlen   = (int)PyArray_DIM(x_array, 0);

    /* Get pointers to the data as C-types. */
    double *X    = (double*)PyArray_DATA(X_array);
    double *x    = (double*)PyArray_DATA(x_array);


    /* Call the external C function to compute the covariance. */
    double *k = covVec(X, x, r, d, nvecs, veclen);



    if ( veclen !=  xlen ) {
    	PyErr_SetString(PyExc_RuntimeError,
    							"Dimensions don't match!!");
		return NULL;
    }

    /* Clean up. */
    Py_DECREF(X_array);
    Py_DECREF(x_array);

    int i;
    for(i = 0 ; i < nvecs ; i++) {
//    	printf("k[%d]   = %f\n",i,k[i]);
		if (k[i] < 0.0) {
			PyErr_SetString(PyExc_RuntimeError,
						"Covariance should be positive but it isn't.");
			return NULL;
		}
    }

	npy_intp dims[1] = {nvecs};

	PyObject *ret = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	memcpy(PyArray_DATA(ret), k, nvecs*sizeof(double));
	free(k);

	return ret;
}






static PyObject *_aux_cov(PyObject *self, PyObject *args)
{
    double r, d;
    PyObject *x_obj, *y_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOdd", &x_obj, &y_obj ,&r, &d ))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    /* If that didn't work, throw an exception. */
    if (x_array == NULL || y_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    /* How many data points are there? */
    int N = (int)PyArray_DIM(x_array, 0);

    /* Get pointers to the data as C-types. */
    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);


    /* Call the external C function to compute the covariance. */
    double value = cov(x, y, r, d, N);

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);


    if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
                    "Covariance should be positive but it isn't.");
        return NULL;
    }

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}
