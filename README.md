Note 2/25/19: Getting error when attempting to run files using gcc, g++, clang, clang++ 

Improved a 2D image convolution implementation to run over 80 times faster. Used a number of performance programming techniques like loop reordering, loop unrolling, SSE intrinsics, parallelization using OpenMP, compiler tricks, branching reduction, and register blocking to speed up the implementation from ~1 Gflop/s to ~80 Gflop/s (average speed over a variety of image sizes).
