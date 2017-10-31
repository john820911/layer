#include <cstdio>
#include <cstdlib>
#include "im_col_convert.h"

void im_col(
    const int channel_im, const int height_im, const int width_im,
    const int ksize, const int stride, const int pad,
    const float *data_im, float *data_col
) {
	int channel_col = channel_im * ksize * ksize;
	int height_col = (height_im + 2 * pad - ksize) / stride + 1;
    int width_col = (width_im + 2 * pad - ksize) / stride + 1;
    for(int c_col = 0; c_col < channel_col; ++c_col) {
        for(int h_col = 0; h_col < height_col; ++h_col) {
            for(int w_col = 0; w_col < width_col; ++w_col) {
            	int c_im = (c_col / ksize) / ksize;
            	int h_im = h_col * stride + (c_col / ksize) % ksize - pad;
                int w_im = w_col * stride + c_col % ksize - pad;
                if (w_im < 0 || h_im < 0 || w_im >= width_im || h_im >= height_im) {
                    data_col[c_col * height_col * width_col + h_col * width_col + w_col] = 0.0;
                }
                else {
                    data_col[c_col * height_col * width_col + h_col * width_col + w_col] = data_im[c_im * height_im * width_im + h_im * width_im + w_im];
                }
            }
        }
    }
}

void col_im(
    const int channel_im, const int height_im, const int width_im,  
    const int ksize, const int stride, const int pad,
    const float *data_col, float *data_im
) {
    int channel_col = channel_im * ksize * ksize;
    int height_col = (height_im + 2 * pad - ksize) / stride + 1;
    int width_col = (width_im + 2 * pad - ksize) / stride + 1;
    for(int c_col = 0; c_col < channel_col; ++c_col) {
        for(int h_col = 0; h_col < height_col; ++h_col) {
            for(int w_col = 0; w_col < width_col; ++w_col) {
                int c_im = (c_col / ksize) / ksize;                
                int h_im = h_col * stride + (c_col / ksize) % ksize - pad;
                int w_im = w_col * stride + c_col % ksize - pad;
                if (w_im < 0 || h_im < 0 || w_im >= width_im || h_im >= height_im) {
                    data_im[c_im * height_im * width_im + h_im * width_im + w_im] += 0.0;
                }
                else {
                	data_im[c_im * height_im * width_im + h_im * width_im + w_im] += data_col[c_col * height_col * width_col + h_col * width_col + w_col];
                }
            }
        }
    }
}
