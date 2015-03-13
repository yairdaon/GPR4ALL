#include <Python.h>
#include <numpy/arrayobject.h>
#include "krigger.h"
#include "aux.h"
#include "g.h"
#include <stdio.h>



static char module_docstring[] =
    "This module provides an interface for calculating g using C.";

static char g_docstring[] =
    "calcualte g and its gradients ";

static char gradKoverSig_docstring[] =
    "calcualte gradient of covariance over variance";

static PyObject *_g(PyObject *self, PyObject *args);
static PyObject *_gradKoverSig(PyObject *self, PyObject *args);


static PyMethodDef module_methods[] = {
	    {"g"       , _g    , METH_VARARGS, g_docstring    },	  
	    {"gradKoverSig"       , _gradKoverSig    , METH_VARARGS, gradKoverSig_docstring    },
	    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC init_g(void) {

    PyObject *m = Py_InitModule3("_g", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}


PyMODINIT_FUNC init_gradKoverSig(void) {

    PyObject *m = Py_InitModule3("_gradKoverSig", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}


























static PyObject *_gradKoverSig(PyObject *self, PyObject *args)
{


	double r, d, reg;
	PyObject *U_obj, *S_obj , *V_obj , *X_obj, *x_obj , *xn_obj;

	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOOOOddd",
			&U_obj, &S_obj, &V_obj,
			&X_obj, &x_obj, &xn_obj,
			&r , &d, &reg ))
		return NULL;

	/* Interpret the input objects as numpy arrays. */
    PyObject *U_array  = PyArray_FROM_OTF(U_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *S_array  = PyArray_FROM_OTF(S_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *V_array  = PyArray_FROM_OTF(V_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array  = PyArray_FROM_OTF(X_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *x_array  = PyArray_FROM_OTF(x_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *xn_array = PyArray_FROM_OTF(xn_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    /* If that didn't work, throw an exception. */
    if (U_array == NULL || S_array == NULL || V_array == NULL
    		|| X_array == NULL || x_array == NULL
    		 || xn_array == NULL) {
        Py_XDECREF(U_array);
        Py_XDECREF(S_array);
        Py_XDECREF(V_array);
        Py_XDECREF(X_array);
        Py_XDECREF(x_array);
        Py_XDECREF(xn_array);
        return NULL;
    }

    /* What are the dimensions? */
    int nvecs  = (int)PyArray_DIM(U_array , 1); //  == nvecs
    int Xlen   = (int)PyArray_DIM(X_array , 0); //  == nvecs 
    int veclen = (int)PyArray_DIM(X_array , 1); //  == veclen 
    int xnlen  = (int)PyArray_DIM(xn_array, 0); //  == veclen
    int Slen   = (int)PyArray_DIM(S_array , 0); //  == nvecs
    int xlen   = (int)PyArray_DIM(x_array , 0); //  == veclen

    if ( Xlen != nvecs ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimension mismatch:  Xarr.shape[0] != U.shape[1].");
    		return NULL;
        }


    if ( xnlen != veclen ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimension mismatch:  Xarr.shape[1] != len(x).");
    		return NULL;
        }



    if ( Slen != nvecs ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimension mismatch:  len(S) != U.shape[1].");
    		return NULL;
        }



    if ( xlen  != veclen ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimension mismatch:  len(x) != X.shape[1].");
    		return NULL;
        }





    /* Get pointers to the data as C-types. */
    double *U    = (double*)PyArray_DATA(U_array);
    double *S    = (double*)PyArray_DATA(S_array);
    double *V    = (double*)PyArray_DATA(V_array);
    double *X    = (double*)PyArray_DATA(X_array);
    double *x    = (double*)PyArray_DATA(x_array);
    double *xn   = (double*)PyArray_DATA(xn_array);


    /* Call the external C function to compute the covariance. */
    double *grad = gradKOverSig(U, S, V, X, x, xn, r, d, reg, nvecs, veclen);

    /* Clean up. */
    Py_DECREF(U_array);
    Py_DECREF(S_array);
    Py_DECREF(V_array);
    Py_DECREF(X_array);
    Py_DECREF(x_array);
    Py_DECREF(xn_array);


    /* Build the output tuple */

    // create gradient of krig python object
    npy_intp dims[1] = {veclen}; // auxilliary array

	PyObject *pyGrad = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	memcpy(PyArray_DATA(pyGrad), grad, veclen*sizeof(double));
	free(grad); // definitely need this


    /* Build the output tuple */
	PyObject *ret = Py_BuildValue("O", pyGrad );
	Py_DECREF(pyGrad);

	return ret;
}




































static PyObject *_g(PyObject *self, PyObject *args)
{


	double r, d, reg;
	PyObject *U_obj, *S_obj , *V_obj , *X_obj, *x_obj , *xn_obj;

	/* Parse the input tuple */
	if (!PyArg_ParseTuple(args, "OOOOOOddd",
			&U_obj, &S_obj, &V_obj,
			&X_obj, &x_obj, &xn_obj,
			&r , &d, &reg ))
		return NULL;

	/* Interpret the input objects as numpy arrays. */
    PyObject *U_array  = PyArray_FROM_OTF(U_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *S_array  = PyArray_FROM_OTF(S_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *V_array  = PyArray_FROM_OTF(V_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array  = PyArray_FROM_OTF(X_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *x_array  = PyArray_FROM_OTF(x_obj , NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *xn_array = PyArray_FROM_OTF(xn_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    /* If that didn't work, throw an exception. */
    if (U_array == NULL || S_array == NULL || V_array == NULL
    		|| X_array == NULL || x_array == NULL
    		 || xn_array == NULL) {
        Py_XDECREF(U_array);
        Py_XDECREF(S_array);
        Py_XDECREF(V_array);
        Py_XDECREF(X_array);
        Py_XDECREF(x_array);
        Py_XDECREF(xn_array);
        return NULL;
    }

    /* What are the dimensions? */
    int nvecs  = (int)PyArray_DIM(U_array , 1); //  == nvecs
    int Xlen   = (int)PyArray_DIM(X_array , 0); //  == nvecs 
    int veclen = (int)PyArray_DIM(X_array , 1); //  == veclen 
    int xnlen  = (int)PyArray_DIM(xn_array, 0); //  == veclen
    int Slen   = (int)PyArray_DIM(S_array , 0); //  == nvecs
    int xlen   = (int)PyArray_DIM(x_array , 0); //  == veclen

    if ( Xlen != nvecs ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimension mismatch:  Xarr.shape[0] != U.shape[1].");
    		return NULL;
        }


    if ( xnlen != veclen ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimension mismatch:  Xarr.shape[1] != len(x).");
    		return NULL;
        }



    if ( Slen != nvecs ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimension mismatch:  len(S) != U.shape[1].");
    		return NULL;
        }



    if ( xlen  != veclen ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimension mismatch:  len(x) != X.shape[1].");
    		return NULL;
        }





    /* Get pointers to the data as C-types. */
    double *U    = (double*)PyArray_DATA(U_array);
    double *S    = (double*)PyArray_DATA(S_array);
    double *V    = (double*)PyArray_DATA(V_array);
    double *X    = (double*)PyArray_DATA(X_array);
    double *x    = (double*)PyArray_DATA(x_array);
    double *xn   = (double*)PyArray_DATA(xn_array);


    /* Call the external C function to compute the covariance. */
    struct gStruct values = g(U, S, V, X, x, xn, r, d, reg, nvecs, veclen);

    /* Clean up. */
    Py_DECREF(U_array);
    Py_DECREF(S_array);
    Py_DECREF(V_array);
    Py_DECREF(X_array);
    Py_DECREF(x_array);
    Py_DECREF(xn_array);


    /* Build the output tuple */

    // unpack the struct
    double * grad    =  values.grad;
    double   gValue  =  values.gValue;
    // free(values); // not sure whether i need to free or not


    // create gradient of krig python object
    npy_intp dims[1] = {veclen}; // auxilliary array

	PyObject *pyGrad = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	memcpy(PyArray_DATA(pyGrad), grad, veclen*sizeof(double));
	free(grad); // definitely need this


    /* Build the output tuple */
	PyObject *ret = Py_BuildValue("dO", gValue, pyGrad );
	Py_DECREF(pyGrad);

	return ret;
}

