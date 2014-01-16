#include <emmintrin.h>
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
#define LOOP_SIZE 8
int conv2D(const float* in, float* out, const int data_size_X, const int data_size_Y,
                    const float* kernel)
{
  // the x coordinate of the kernel's center
  const int kern_cent_X = (KERNX - 1)/2;
  // the y coordinate of the kernel's center
  const int kern_cent_Y = (KERNY - 1)/2;
  omp_set_num_threads(10);
  if (data_size_X > 10) {
    // the last value of x which we can vectorize without running off the edge
    // of the matrix
    const int x_lim = (data_size_X-LOOP_SIZE-kern_cent_X-(kern_cent_X%LOOP_SIZE))/LOOP_SIZE*LOOP_SIZE+(kern_cent_X%LOOP_SIZE);
    // main convolution loop
#pragma omp parallel for schedule(dynamic, 8)
    for(int y = 0; y < data_size_Y; y++){ // the y coordinate of the output location we're focusing on
      for(int x = 0; x < kern_cent_X; x++) {
        float value = 0.0;
        for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
          if(x+i<0 || x+i>=data_size_X) {
            continue;
          }
          for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
            // only do the operation if not out of bounds
            if(y+j>-1 && y+j<data_size_Y){
              //Note that the kernel is flipped
              value += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
            }
          }
        }
        out[x+y*data_size_X] += value;
      }
        __m128 vec1;
        __m128 result1;
        __m128 result2;
        __m128 ker;

      for(int x = kern_cent_X; x <= x_lim; x+=LOOP_SIZE){ // the x coordinate of the output location we're focusing on
        // load 4 adjacent values from in into vec - for some reason this increases cache performance
        vec1 = _mm_loadu_ps(in + x + y*data_size_X);
        result1 = _mm_setzero_ps();
        result2 = _mm_setzero_ps();
        for(int j = -kern_cent_Y; j <= kern_cent_Y; j++) {
          if(y + j < 0 || y + j >= data_size_Y) {
            continue;
          }
          for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
            ker = _mm_load_ps1(kernel + kern_cent_X - i + (kern_cent_Y - j)*KERNX);
            result1 = _mm_add_ps(result1, _mm_mul_ps(ker, _mm_loadu_ps(in + x + i + (y+j)*data_size_X)));
            result2 = _mm_add_ps(result2, _mm_mul_ps(ker, _mm_loadu_ps(in + x + i + (y+j)*data_size_X + 4)));
          }
        }
        _mm_storeu_ps(out + x + y*data_size_X, result1);
        _mm_storeu_ps(out + x + y*data_size_X + 4, result2);
      }
      for(int x = x_lim+LOOP_SIZE; x < data_size_X; x++) {
        float value = 0.0;
        for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
          for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
            // only do the operation if not out of bounds
            if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
              //Note that the kernel is flipped
              value += kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
            }
          }
        }
        out[x+y*data_size_X] += value;
      }
    }
    return 1;
  }
#pragma omp parallel for
  for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
    for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
      for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
        for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
          // only do the operation if not out of bounds
          if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
            //Note that the kernel is flipped
            out[x+y*data_size_X] += 
                kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
          }
        }
      }
    }
  }
  return 1;
}
