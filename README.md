# layer
### An easy implementation of convolutional/deconvolutional layer using im_col_convert and gemm API.
### For calling the convolutional/deconvolutional layer, you should define yours:
    1. input_shape -- 1D tensor of shape [ batch, channel, height, width ]
    2. input -- 1D tensor of data stored in format [ batch * channel * height * width ]
    3. filter_shape -- 1D tensor of shape [ channel_out, channel_in, height, width ]
    4. filter -- 1D tensor of data stored in format [ channel_out * channel_in * height * width ]
    5. strides -- 1D tensor of shape [ 1, stride, stride, 1 ]
    6. padding -- integer of padding length at each side
    7. output_shape -- 1D tensor of shape [ batch, channel, height, width ]
    8. output -- 1D tensor of data stored in format [ batch * channel * height * width ]
### compile: g++ -std=c++11 gemm.cpp im_col_convert.cpp layer.cpp main.cpp -o main
### execution: ./main
### test: python test.py
