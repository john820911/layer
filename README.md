# layer
An easy implementation of convolutional/deconvolutional layer using im_col_convert and gemm API.<br/>
For calling the convolutional/deconvolutional layer, you should define yours:<br/>
    >1. input_shape -- 1D tensor of shape [ batch, channel, height, width ]<br/>
    >2. input -- 1D tensor of data stored in format [ batch * channel * height * width ]<br/>
    >3. filter_shape -- 1D tensor of shape [ channel_out, channel_in, height, width ]<br/>
    >4. filter -- 1D tensor of data stored in format [ channel_out * channel_in * height * width ]<br/>
    >5. strides -- 1D tensor of shape [ 1, stride, stride, 1 ]<br/>
    >6. padding -- integer of padding length at each side<br/>
    >7. output_shape -- 1D tensor of shape [ batch, channel, height, width ]<br/>
    >8. output -- 1D tensor of data stored in format [ batch * channel * height * width ]<br/>
compile: g++ -std=c++11 gemm.cpp im_col_convert.cpp layer.cpp main.cpp -o main<br/>
execution: ./main<br/>
test: python test.py<br/>
