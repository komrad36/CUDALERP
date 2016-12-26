The file CUDALERP.h exposes two extremely high performance GPU
resize operations,
CUDALERP (bilinear interpolation), and 
CUDANERP (nearest neighbor interpolation).

CUDALERP offers superior accuracy to CUDA's built-in texture
interpolator at comparable performance. The accuracy if compiled
with -use-fast-math off is nearly equivalent to my CPU interpolator,
KLERP, while still being as fast as the built-in interpolation.

Particularly for large images, CUDALERP dramatically outperforms
even the highly tuned CPU AVX2 versions.

All functionality is contained in the header 'CUDALERP.h' and
the source file 'CUDALERP.cu' and has no external dependencies at all.

Note that these are intended for computer vision use(hence the speed)
and are designed for grayscale images.

The file 'main.cpp' is an example and speed test driver.
