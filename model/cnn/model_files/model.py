import numpy as np
import tensorflow as tf
from get_images_for_training import get_features_and_labels
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.python import saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

features, labels = get_features_and_labels()
# print(labels)
X_Train_features, X_Test_features, Y_Train_features, Y_Test_features = train_test_split(
    features, labels, test_size=0, shuffle=True)


learning_rate = 0.001
max_pool_size = 2

image_height = features.shape[1]
image_width = features.shape[2]
num_channels = features.shape[3]
filter_size = 4
conv_1_features = 32
conv_2_features = 64
target_size = 2
hidden_layer_1_output_size = 100
hidden_layer_2_output_size = 2

real_example = tf.placeholder(dtype=tf.float32, shape=[
                              1, image_height, image_width, num_channels], name="imageToPredict")
conv_1_filter = tf.get_variable("conv1_filter", initializer=tf.random.truncated_normal(
    shape=(filter_size, filter_size, num_channels, conv_1_features)))
conv_1_bias = tf.get_variable("conv1_bias", initializer=tf.zeros(
    shape=(conv_1_features)))

conv_2_filter = tf.get_variable('conv2_filter', initializer=tf.random.truncated_normal(
    shape=(filter_size, filter_size, conv_1_features, conv_2_features)))
conv_2_bias = tf.get_variable("conv2_bias", initializer=tf.zeros(
    shape=(conv_2_features)))

reduced_image_height = image_height // pow(max_pool_size, 2)
reduced_image_width = image_width // pow(max_pool_size, 2)

fully_connected_input_size = reduced_image_width * \
    reduced_image_height * conv_2_features

hidden_layer_1_weights = tf.get_variable("hidden_layer_1_weights", initializer=tf.random.truncated_normal(
    shape=(fully_connected_input_size, hidden_layer_1_output_size)))
hidden_layer_1_bias = tf.get_variable('hidden_layer_1_bias', initializer=tf.zeros(
    shape=hidden_layer_1_output_size))

hidden_layer_2_weights = tf.get_variable('hidden_layer_2_weights', initializer=tf.Variable(tf.random.truncated_normal(
    shape=(hidden_layer_1_output_size, hidden_layer_2_output_size))))
hidden_layer_2_bias = tf.get_variable('hidden_layer_2_bias', initializer=tf.zeros(
    shape=hidden_layer_2_output_size))

batch_size = X_Train_features.shape[0]
test_size = X_Test_features.shape[0]

X_Train = tf.placeholder(dtype=tf.float32, shape=(
    batch_size, image_width, image_height, num_channels))
Y_Train = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])

# X_Test = tf.placeholder(dtype=tf.float32, shape=(
#     test_size, image_width, image_height, num_channels))
# Y_Test = tf.placeholder(dtype=tf.float32, shape=[test_size, 2])


def get_conv(input_data):
    conv1 = tf.nn.conv2d(input_data, conv_1_filter, [
                         1, 1, 1, 1], padding="SAME")
    relu_1 = tf.nn.relu(tf.nn.bias_add(conv1, conv_1_bias))
    maxpool_1_output = tf.nn.max_pool(relu_1, ksize=[
        1, max_pool_size, max_pool_size, 1], strides=[1, max_pool_size, max_pool_size, 1], padding="SAME")

    conv2 = tf.nn.conv2d(maxpool_1_output, conv_2_filter,
                         [1, 1, 1, 1], padding="SAME")
    relu_2 = tf.nn.relu(tf.nn.bias_add(conv2, conv_2_bias))
    maxpool_2_output = tf.nn.max_pool(relu_2, ksize=[
        1, max_pool_size, max_pool_size, 1], strides=[1, max_pool_size, max_pool_size, 1], padding="SAME")

    maxpool_out_shape = maxpool_2_output.get_shape().as_list()
    final_output_size = maxpool_out_shape[1] * \
        maxpool_out_shape[2] * maxpool_out_shape[3]

    maxpool_out_flatten = tf.reshape(maxpool_2_output, shape=[
        maxpool_out_shape[0], final_output_size])

    fully_connected_1_output = tf.nn.relu(tf.add(
        tf.matmul(maxpool_out_flatten, hidden_layer_1_weights), hidden_layer_1_bias))

    output = tf.add(tf.matmul(fully_connected_1_output,
                              hidden_layer_2_weights), hidden_layer_2_bias)
    return output


train_output = get_conv(X_Train)
# test_output = get_conv(X_Test)
real_output = get_conv(real_example)

train_predicted = tf.nn.softmax(train_output)
train_accuracy = tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(train_predicted, axis=1), tf.argmax(Y_Train, axis=1)), tf.float32))

# test_predicted = tf.nn.softmax(test_output)
# test_accuracy = tf.reduce_mean(tf.cast(
#     tf.equal(tf.argmax(test_predicted, axis=1), tf.argmax(Y_Test, axis=1)), tf.float32))

predited_output_for_real_world = tf.nn.softmax(real_output, name="label")

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y_Train, logits=train_output))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


init = tf.global_variables_initializer()


epochs = 10
iterations = 10
with tf.Session() as sess:
    sess.run(init)
    for e in range(epochs):
        for i in range(iterations):
            sess.run(optimizer, feed_dict={
                     X_Train: X_Train_features, Y_Train: Y_Train_features})
        loss_value = sess.run(
            loss, feed_dict={X_Train: X_Train_features, Y_Train: Y_Train_features})
        accuracy_value_Train = sess.run(
            train_accuracy, feed_dict={X_Train: X_Train_features, Y_Train: Y_Train_features})
        print("Train_accuracy: ", accuracy_value_Train)

    # print("Test Accuracy: ", sess.run(
    #     test_accuracy, feed_dict={X_Test: X_Test_features, Y_Test: Y_Test_features}))

    builder = saved_model.builder.SavedModelBuilder("../saved_model")
    signature = predict_signature_def(inputs={'imageToPredict': real_example}, outputs={
                                      "label": predited_output_for_real_world})
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=["serve"],
                                         signature_def_map={'predict': signature})
    builder.save()
