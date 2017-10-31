#include <cstdio>
#include <cstdlib>
#include "gemm.h"
#include "im_col_convert.h"
#include "layer.h"

void convolutional_layer(
    const int *input_shape,
    const float *input,
    const int *filter_shape,
    const float *filter,
    const int *strides,
    const int padding,
    const int *output_shape,
    float *output
) {
    for(int input_batch = 0; input_batch < input_shape[0]; ++input_batch) {
        const float *input_data = input + input_batch * input_shape[1] * input_shape[2] * input_shape[3];
        float *temp_data = (float *)malloc((filter_shape[1] * filter_shape[2] * filter_shape[3] * output_shape[2] * output_shape[3]) * sizeof(float));
        for(int i = 0; i < filter_shape[1] * filter_shape[2] * filter_shape[3] * output_shape[2] * output_shape[3]; ++i) {
            temp_data[i] = 0.0;
        }        
        float *output_data = output + input_batch * output_shape[1] * output_shape[2] * output_shape[3];
        im_col(
            input_shape[1], input_shape[2], input_shape[3],
            filter_shape[2], strides[1], padding,
            input_data, temp_data
        );
        gemm_nn(
            filter_shape[0], output_shape[2] * output_shape[3], filter_shape[1] * filter_shape[2] * filter_shape[3],
            filter, filter_shape[1] * filter_shape[2] * filter_shape[3],
            temp_data, output_shape[2] * output_shape[3],
            output_data, output_shape[2] * output_shape[3]
        );
        free(temp_data);
    }
}

void deconvolutional_layer(
    const int *input_shape,
    const float *input,
    const int *filter_shape,
    const float *filter,
    const int *strides,
    const int padding,
    const int *output_shape,
    float *output    
) {
    for(int input_batch = 0; input_batch < input_shape[0]; ++input_batch) {
        const float *input_data = input + input_batch * input_shape[1] * input_shape[2] * input_shape[3];
        float *temp_data = (float *)malloc((filter_shape[1] * filter_shape[2] * filter_shape[3] * input_shape[2] * input_shape[3]) * sizeof(float));
        for(int i = 0; i < filter_shape[1] * filter_shape[2] * filter_shape[3] * input_shape[2] * input_shape[3]; ++i) {
            temp_data[i] = 0.0;
        }
        float *output_data = output + input_batch * output_shape[1] * output_shape[2] * output_shape[3];
        gemm_tn(
            filter_shape[1] * filter_shape[2] * filter_shape[3], input_shape[2] * input_shape[3], filter_shape[0],
            filter, filter_shape[1] * filter_shape[2] * filter_shape[3],
            input_data, input_shape[2] * input_shape[3],
            temp_data, input_shape[2] * input_shape[3]
        );
        col_im(
            output_shape[1], output_shape[2], output_shape[3],
            filter_shape[2], strides[1], padding,
            temp_data, output_data
        );
        free(temp_data);
    }    
}

void test_convolutional_layer() {
    int input_shape[] = { 1, 3, 5, 5 };
    float *input = (float *)malloc((input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]) * sizeof(float));
    for(int i = 0; i < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]; ++i) {
        input[i] = float(i);
    }
    int filter_shape[] = { 3, 3, 3, 3 };
    float *filter = (float *)malloc((filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]) * sizeof(float));
    for(int i = 0; i < filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]; ++i) {
        filter[i] = float(i);
    }
    int strides[] = { 1, 1, 1, 1 };
    int padding = 0;
    int output_shape[] = { 1, 3, 3, 3 };
    float *output = (float *)malloc((output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]) * sizeof(float));
    for(int i = 0; i < output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]; ++i) {
        output[i] = 0.0;
    }
    convolutional_layer(
        input_shape,
        input,
        filter_shape,
        filter,
        strides,
        padding,
        output_shape,
        output
    );
    printf("Output of convolutional layer:\n");
    for(int i = 0; i < output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]; ++i) {
        printf("%f ", output[i]);
    }
    printf("\n");
    free(input);
    free(filter);
    free(output);
}

void test_deconvolutional_layer() {
    int input_shape[] = { 1, 3, 3, 3 };
    float *input = (float *)malloc((input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]) * sizeof(float));
    for(int i = 0; i < input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]; ++i) {
        input[i] = float(i);
    }
    int filter_shape[] = { 3, 3, 3, 3 };
    float *filter = (float *)malloc((filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]) * sizeof(float));
    for(int i = 0; i < filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]; ++i) {
        filter[i] = float(i);
    }
    int strides[] = { 1, 1, 1, 1 };
    int padding = 0;
    int output_shape[] = { 1, 3, 5, 5 };
    float *output = (float *)malloc((output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]) * sizeof(float));
    for(int i = 0; i < output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]; ++i) {
        output[i] = 0.0;
    }
    deconvolutional_layer(
        input_shape,
        input,
        filter_shape,
        filter,
        strides,
        padding,
        output_shape,
        output
    );
    printf("Output of deconvolutional layer:\n");
    for(int i = 0; i < output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]; ++i) {
        printf("%f ", output[i]);
    }
    printf("\n");
    free(input);
    free(filter);
    free(output);
}
