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

@cython.boundscheck(False)
@cython.wraparound(False)
def gradient_helper(double[:, :] background, double a2, double a1, double a0, double b2, double b1):

    cdef double offset
    cdef int rows
    cdef int cols
    cdef double rd
    cdef double cd
    cdef double[:, :] cdata

    cdata = background

    rows, cols = background.shape[0:2]

    for r in range(rows):
        rd = float(r)
        for c in range(cols):
            cd = float(c)
            offset = a2*cd**2 + a1*cd + a0
            cdata[r, c] = b2*rd**2 + b1*rd + offset

@cython.boundscheck(False)
@cython.wraparound(False)
def simulate_helper(uint8[:, :, :] img, uint8[:, :] player_colors, uint16[:, :] active_points_x, uint16[:, :] active_points_y, uint16[:, :] active_points_x2, uint16[:, :] active_points_y2):

    cdef int keep_going = 1
    cdef int dxx[8]
    cdef int dyy[8]
    cdef int dx 
    cdef int dy 
    cdef int n_active_points[100] # assumes you'll never have more than 100 players  
    cdef int n2_active_points[100]
    cdef uint8[:, :, :] cimg = img
    cdef uint8[:, :] cplayer_colors = player_colors
    cdef uint16[:, :] cactive_points_x = active_points_x
    cdef uint16[:, :] cactive_points_y = active_points_y
    cdef uint16[:, :] cactive_points_x2 = active_points_x2
    cdef uint16[:, :] cactive_points_y2 = active_points_y2
    cdef int n_players
    cdef int n 
    cdef int a 
    cdef int p 
    cdef int k
    cdef int q
    cdef int point[2]
    cdef int pixel[3]

    n_players = player_colors.shape[0]

    # initial number of active points for each player
    for n in range(n_players):
        n_active_points[n] = 1

    dxx[:] = [1, 1, 0, -1, -1, -1, 0, 1]
    dyy[:] = [0, 1, 1, 1, 0, -1, -1, -1]

    while keep_going > 0:
        keep_going = -1
        for p in range(n_players): # a particular player
            n2_active_points[p] = 0 # new active points TBD
            for a in range(n_active_points[p]): # number of active points for this player
                keep_going = 1
                point[0] = cactive_points_x[a, p]
                point[1] = cactive_points_y[a, p]
                for k in range(8):
                    dx = dxx[k]
                    dy = dyy[k]
                    for q in range(3):
                        pixel[q] = cimg[point[0] + dx, point[1] + dy, q]
                    if pixel[0] == 200 and pixel[1] == 200 and pixel[2] == 200:
                        cactive_points_x2[n2_active_points[p], p] = point[0] + dx
                        cactive_points_y2[n2_active_points[p], p] = point[1] + dy
                        for q in range(3):
                            cimg[point[0] + dx, point[1] + dy, q] = player_colors[p, q]
                        n2_active_points[p] += 1
            n_active_points[p] = n2_active_points[p]
            for a in range(n2_active_points[p]):
                cactive_points_x[a, p] = cactive_points_x2[a, p]
                cactive_points_y[a, p] = cactive_points_y2[a, p]
