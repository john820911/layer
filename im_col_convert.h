#ifndef IM_COL_CONVERT_H
#define IM_COL_CONVERT_H

void im_col(
    const int channel_im, const int height_im, const int width_im, 
    const int ksize, const int stride, const int pad,
    const float *data_im, float *data_col
);

void col_im(
    const int channel_im, const int height_im, const int width_im,
    const int ksize, const int stride, const int pad,
    const float *data_col, float *data_im
);

#endif
