#include <Python.h>
#include <numpy/arrayobject.h>
#include "krigger.h"
#include "aux.h"
#include <stdio.h>



static char module_docstring[] =
    "This module provides an interface for doing krigging using C.";

static char krig_docstring[] =
    "Do kriging with no variance or gradients ";

static char krig_var_docstring[] =
    "Do kriging with variance but without gradients ";

static char krig_grads_docstring[] =
    "Do kriging with variance and gradients ";

static PyObject *_krigger_krig(PyObject *self, PyObject *args);

static PyObject *_krigger_krigVar(PyObject *self, PyObject *args);

static PyObject *_krigger_krigGrads(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
	    {"krig"     , _krigger_krig   , METH_VARARGS, krig_docstring    },
	    {"krig_var"       , _krigger_krigVar    , METH_VARARGS, krig_var_docstring    },
	    {"krig_grads"       , _krigger_krigGrads    , METH_VARARGS, krig_grads_docstring    },
	    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC init_krigger(void) {

    PyObject *m = Py_InitModule3("_krigger", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}







static PyObject *_krigger_krigGrads(PyObject *self, PyObject *args)
{


    double prior, r, d, reg;
	PyObject *U_obj, *S_obj , *V_obj , *X_obj, *x_obj , *y_obj, *g_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOOOOdddd",
    				&U_obj, &S_obj, &V_obj,
					&X_obj, &x_obj, &y_obj, &g_obj,
					&prior, &r , &d, &reg ))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *U_array = PyArray_FROM_OTF(U_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *V_array = PyArray_FROM_OTF(V_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *g_array = PyArray_FROM_OTF(g_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    /* If that didn't work, throw an exception. */
    if (U_array == NULL || S_array == NULL || V_array == NULL
    		|| X_array == NULL || x_array == NULL
    		 || y_array == NULL || g_array == NULL) {
        Py_XDECREF(U_array);
        Py_XDECREF(S_array);
        Py_XDECREF(V_array);
        Py_XDECREF(X_array);
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(g_array);
        return NULL;
    }

    /* What are the dimensions? */
    int nvecs  = (int)PyArray_DIM(U_array, 1);
    int xlen   = (int)PyArray_DIM(x_array, 0);
    int veclen = (int)PyArray_DIM(X_array, 1);
    int ylen   = (int)PyArray_DIM(y_array, 0);
    int slen   = (int)PyArray_DIM(S_array, 0);
    int glen   = (int)PyArray_DIM(g_array, 0);

    if ( (nvecs!=ylen) || (xlen!=veclen)
    		 || (slen!=nvecs)|| (glen!=veclen) ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimensions don't match!!");
    		return NULL;
        }

    /* Get pointers to the data as C-types. */
    double *U    = (double*)PyArray_DATA(U_array);
    double *S    = (double*)PyArray_DATA(S_array);
    double *V    = (double*)PyArray_DATA(V_array);
    double *X    = (double*)PyArray_DATA(X_array);
    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);
    double *g    = (double*)PyArray_DATA(g_array);


    /* Call the external C function to compute the covariance. */
    struct krigStruct values = krigGrads(U, S, V, X, x, y, g,
    			prior, r, d, reg, nvecs, veclen);

    /* Clean up. */
    Py_DECREF(U_array);
    Py_DECREF(S_array);
    Py_DECREF(V_array);
    Py_DECREF(X_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(g_array);


    /* Build the output tuple */

    // unpack the struct
    double * gradKrig = values.gradKrig;
    double * gradSig2 = values.gradSig2;
    double krig       = values.krig;
    double sigSqr     = values.sigSqr;
//    double * other    = values.other;

    // create gradient of krig python object
    npy_intp dims[1] = {veclen}; // auxilliary array

	PyObject *pyGradKrig = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	memcpy(PyArray_DATA(pyGradKrig), gradKrig, veclen*sizeof(double));
	free(gradKrig);


	PyObject *pyGradSig2 = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
	memcpy(PyArray_DATA(pyGradSig2), gradSig2, veclen*sizeof(double));
    free(gradSig2);

    /* Build the output tuple */
	PyObject *ret = Py_BuildValue("ddOO", krig, sigSqr, pyGradKrig, pyGradSig2 );
	Py_DECREF(pyGradKrig);
	Py_DECREF(pyGradSig2);

	return ret;
}























static PyObject *_krigger_krigVar(PyObject *self, PyObject *args)
{


    double prior, r, d, reg;
	PyObject *U_obj, *S_obj , *V_obj , *X_obj, *x_obj , *y_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOOOdddd",
    				&U_obj, &S_obj, &V_obj,
					&X_obj, &x_obj, &y_obj,
					&prior, &r , &d, &reg ))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *U_array = PyArray_FROM_OTF(U_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *V_array = PyArray_FROM_OTF(V_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    /* If that didn't work, throw an exception. */
    if (U_array == NULL || S_array == NULL || V_array == NULL
    		|| X_array == NULL || x_array == NULL || y_array == NULL) {
        Py_XDECREF(U_array);
        Py_XDECREF(S_array);
        Py_XDECREF(V_array);
        Py_XDECREF(X_array);
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    /* What are the dimensions? */
    int nvecs  = (int)PyArray_DIM(U_array, 0);
    int xlen   = (int)PyArray_DIM(x_array, 0);
    int veclen = (int)PyArray_DIM(X_array, 1);
    int ylen   = (int)PyArray_DIM(y_array, 0);
    int slen   = (int)PyArray_DIM(S_array, 0);


    if ( (nvecs!=ylen) || (xlen!=veclen) || (slen!=nvecs) ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimensions don't match!!");
    		return NULL;
        }

    /* Get pointers to the data as C-types. */
    double *U    = (double*)PyArray_DATA(U_array);
    double *S    = (double*)PyArray_DATA(S_array);
    double *V    = (double*)PyArray_DATA(V_array);
    double *X    = (double*)PyArray_DATA(X_array);
    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);


    /* Call the external C function to compute the covariance. */
    double * values = krigVar(U, S, V, X, x, y,
    			prior, r, d, reg, nvecs, veclen);

    /* Clean up. */
    Py_DECREF(U_array);
    Py_DECREF(S_array);
    Py_DECREF(V_array);
    Py_DECREF(X_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);


    /* Build the output tuple */

    double krig = values[0];
    double var  = values[1];
    free(values);

    /* Build the output tuple */
	PyObject *ret = Py_BuildValue("dd", krig, var);
	return ret;

}


static PyObject *_krigger_krig(PyObject *self, PyObject *args)
{


    double prior, r, d, reg;
	PyObject *U_obj, *S_obj , *V_obj , *X_obj, *x_obj , *y_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OOOOOOdddd",
    				&U_obj, &S_obj, &V_obj,
					&X_obj, &x_obj, &y_obj,
					&prior, &r , &d, &reg ))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *U_array = PyArray_FROM_OTF(U_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *V_array = PyArray_FROM_OTF(V_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *X_array = PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);


    /* If that didn't work, throw an exception. */
    if (U_array == NULL || S_array == NULL || V_array == NULL
    		|| X_array == NULL || x_array == NULL || y_array == NULL) {
        Py_XDECREF(U_array);
        Py_XDECREF(S_array);
        Py_XDECREF(V_array);
        Py_XDECREF(X_array);
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        return NULL;
    }

    /* What are the dimensions? */
    int nvecs  = (int)PyArray_DIM(U_array, 0);
    int xlen   = (int)PyArray_DIM(x_array, 0);
    int veclen = (int)PyArray_DIM(X_array, 1);
    int ylen   = (int)PyArray_DIM(y_array, 0);
    int slen   = (int)PyArray_DIM(S_array, 0);


    if ( (nvecs!=ylen) || (xlen!=veclen) || (slen!=nvecs) ) {
        	PyErr_SetString(PyExc_RuntimeError,
        							"Dimensions don't match!!");
    		return NULL;
        }

    /* Get pointers to the data as C-types. */
    double *U    = (double*)PyArray_DATA(U_array);
    double *S    = (double*)PyArray_DATA(S_array);
    double *V    = (double*)PyArray_DATA(V_array);
    double *X    = (double*)PyArray_DATA(X_array);
    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);


    /* Call the external C function to compute the covariance. */
    double value = krig(U, S, V, X, x, y,
    			prior, r, d, reg, nvecs, veclen);
    /* Clean up. */
    Py_DECREF(U_array);
    Py_DECREF(S_array);
    Py_DECREF(V_array);
    Py_DECREF(X_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);


    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}

