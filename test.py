import pprint
import numpy as np
import tensorflow as tf

# test convolutional layer
input_feature_map = np.arange(75).reshape(1, 3, 5, 5)
input_feature_map = np.transpose(input_feature_map, (0, 2, 3, 1))
filter_ = np.arange(81).reshape(3, 3, 3, 3)
filter_ = np.transpose(filter_, (2, 3, 1, 0))

input_feature_map = tf.constant(
	value=input_feature_map,
	dtype=tf.float32,
	shape=(1, 5, 5, 3),
	name="input_feature_map"
)
filter_ = tf.constant(
	value=filter_,
	dtype=tf.float32,
	shape=(3, 3, 3, 3),
	name="filter"
)
output_feature_map = tf.nn.conv2d(
	input=input_feature_map,
	filter=filter_,
	strides=(1, 1, 1, 1),
	padding="VALID",
	use_cudnn_on_gpu=True,
	data_format="NHWC",
	name="output_feature_map"
)

session = tf.Session()
pp = pprint.PrettyPrinter()
print "Output of convolutional layer:"
pp.pprint(session.run(output_feature_map).tolist())

# test deconvolutional layer
output_feature_map = np.arange(27).reshape(1, 3, 3, 3)
output_feature_map = np.transpose(output_feature_map, (0, 2, 3, 1))
filter_ = np.arange(81).reshape(3, 3, 3, 3)
filter_ = np.transpose(filter_, (2, 3, 1, 0))

output_feature_map = tf.constant(
	value=output_feature_map,
	dtype=tf.float32,
	shape=(1, 3, 3, 3),
	name="output_feature_map"
)
filter_ = tf.constant(
	value=filter_,
	dtype=tf.float32,
	shape=(3, 3, 3, 3),
	name="filter_"
)
input_feature_map = tf.nn.conv2d_transpose(
	value=output_feature_map,
	filter=filter_,
	output_shape=(1, 5, 5, 3),
	strides=(1, 1, 1, 1),
	padding="VALID",
	data_format="NHWC",
	name="input_feature_map"
)

session = tf.Session()
pp = pprint.PrettyPrinter()
print "Output of deconvolutional layer:"
pp.pprint(session.run(input_feature_map).tolist())
