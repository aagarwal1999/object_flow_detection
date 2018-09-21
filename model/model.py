import tensorflow as tf

# getting the data
data
labels

# hyperparameters and configs
batch_size = 1
layer_dims = [10, 5, 3] #0th elem is input layer, last elem is prediction layer
layer_weights_biases = []

# create the variables to train on
layer_dim_pairs = zip(layers_dims[:-1], layers_dims[1:])
for layer_dim0, layer_dim1 in layer_dim_pairs:
    layer_weights = tf.Variable(tf.float32, shape=(layer_dim0, layer_dim1))
    layer_biases = tf.Variable(tf.float32, shape=(batch_size, layer_dim1))
    layer_weights_biases.append((layer_weights, layer_biases))

# connecting the layers
input_dim = layer_dims[0]
input_layer = tf.placeholder(tf.float32, shape=(batch_size, input_dim))

layer = input_layer
for weights, biases in layer_weights_biases:
    layer = tf.matmul(layer, weights) + biases
output_layer = layer

# loss fn

# split output prediction, since we need to perform
# binary classificaion and movement regression
is_consec_outputs = output_layer[:, 0] # first col contains the probability
movement_outputs = output_layer[:, 1:] # second col contains the dx, dy

cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits()
regression_loss = tf.losses.mean_squared_error()
regularization_loss =
loss = cross_entropy_loss + regression_loss + regularization_loss


with tf.Session() as sess:

