'''

whenever you change this file... --> just run compile_pyxes.py

or manually...
cd /home/dan/Dropbox/'Shared Dropbox'/Work/Projects/temperature_monitor
python setup.py build_ext --inplace

Cython uses the normal C syntax for C types, including pointers.
It provides all the standard C types, namely char, short, int, long, long long as well as their unsigned versions, e.g. unsigned int.
The special bint type is used for C boolean values (int with 0/non-0 values for False/True) and Py_ssize_t for (signed) sizes of Python containers.

NumPy dtype          Numpy Cython type         C Cython type identifier

np.bool_             None                      None
np.int_              cnp.int_t                 long
np.intc              None                      int
np.intp              cnp.intp_t                ssize_t
np.int8              cnp.int8_t                signed char
np.int16             cnp.int16_t               signed short
np.int32             cnp.int32_t               signed int
np.int64             cnp.int64_t               signed long long
np.uint8             cnp.uint8_t               unsigned char
np.uint16            cnp.uint16_t              unsigned short
np.uint32            cnp.uint32_t              unsigned int
np.uint64            cnp.uint64_t              unsigned long
np.float_            cnp.float64_t             double
np.float32           cnp.float32_t             float
np.float64           cnp.float64_t             double
np.complex_          cnp.complex128_t          double complex
np.complex64         cnp.complex64_t           float complex
np.complex128        cnp.complex128_t          double complex

'''

import numpy as np
cimport numpy as cnp
cimport cython
ctypedef cnp.uint8_t uint8
ctypedef cnp.uint16_t uint16
ctypedef cnp.uint64_t uint64
#ctypedef unsigned long ULong
ctypedef cnp.int32_t int32
ctypedef cnp.float64_t float64
ctypedef cnp.float32_t float32
from libcpp cimport bool
from libc.math cimport log, floor, INFINITY, ceil, round
from cython.view cimport array
from cpython cimport array as carray
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
def remove_skeletal_spikes(uint8[:, :, :] skeleton, uint8[:, :, :] img):

    cdef uint8[:, :, :] cdata
    cdef uint8[:, :, :] cimg

    cdata = skeleton
    cimg = img

    cdef int rows
    cdef int cols
    cdef int count
    cdef int r
    cdef int c
    cdef int x
    cdef int y

    #cdef carray.array a = array.array('i', [-1, 0, 1])
    cdef int i2[3]
    i2[:] = [-1, 0, 1]

    rows, cols = skeleton.shape[0:2]

    for r in range(rows):
        for c in range(cols):
            if cdata[r, c, 1] == 255:
                count = 0
                for x in i2:
                    for y in i2:
                        if x == 0 and y == 0:
                            continue
                        if cdata[r+x, c+y, 1] == 255:
                            count += 1

                # additionally, turn black if a surrounding img pixel is black
                for x in i2:
                    for y in i2:
                        if x == 0 and y == 0:
                            continue
                        if cimg[r+x, c+y, 1] == 0:
                            cdata[r, c, 0] = 0
                            cdata[r, c, 1] = 0
                            cdata[r, c, 2] = 0

                if count == 1:
                    cdata[r, c, 0] = 0
                    cdata[r, c, 1] = 0
                    cdata[r, c, 2] = 0
                    skeleton = remove_skeletal_spikes(skeleton, img)

    return skeleton

@cython.boundscheck(False)
@cython.wraparound(False)
def draw_skeleton(uint8[:, :, :] img, uint8[:, :, :] skeleton):

    cdef uint8[:, :, :] cimg
    cdef uint8[:, :, :] cskeleton

    cdef int rows
    cdef int cols

    cimg = img
    cskeleton = skeleton

    rows, cols = img.shape[0:2]

    for r in range(rows):
        for c in range(cols):
            if cskeleton[r, c, 1] == 255:
                cimg[r, c, 0] = 255
                cimg[r, c, 1] = 0
                cimg[r, c, 2] = 0

    return img