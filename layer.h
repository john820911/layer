#ifndef LAYER_H
#define LAYER_H

void convolutional_layer(
    const int *input_shape,
    const float *input,
    const int *filter_shape,
    const float *filter,
    const int *strides,
    const int padding,
    const int *output_shape,
    float *output
);

void deconvolutional_layer(
    const int *input_shape,
    const float *input,
    const int *filter_shape,
    const float *filter,
    const int *strides,
    const int padding,
    const int *output_shape,
    float *output
);

void test_convolutional_layer();

void test_deconvolutional_layer();

#endif
