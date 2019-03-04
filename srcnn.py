"""
ECS795P Deep Learning and Computer Vision
Coursework 1: SRCNN
Dimitrios Stoidis ID: 180933714

Important note:
When running this code, 5 images will appear:
first the weights of layer 1 then layer 2 followed by layer 3.
Next the feature maps of convolutional layer 1 and 2 will appear consecutively
If you do not want them to appear please comment the relevant lines:
visualise_w1()
visualise_w2()
visualise_w3()
visualise_conv1()
visualise_conv2()

Also the lines for saving the relevant images are commented so that they are not saved on your PC.

Finally, when running the code, the following should be printed:
    the values for the first 9x9 filter weights in the 1st conv layer
    the bias of the 10th filter in 1st conv layer
    the 5th filter weight value of 2nd conv layer
    the bias of the 6th filter of the 2nd conv layer
    the values for the 1st (5x5) filter weights in the 3rd conv layer
    the bias of the filter in 3rd conv layer
    the size of the super-resolution image
    the size of the cropped ground truth image
    the PSNR of the super-resolution image wrt the cropped ground truth image
    the PSNR value of the blurred image wrt the ground truth image
"""



import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import scipy
import skimage
from skimage.transform import resize
import pdb


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def preprocess(path, scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)

  label_ = modcrop(image, scale)

  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.

  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

  return input_, label_

"""Set the image hyper parameters
"""
# channel dimension set to 1 for grayscale image and input size to 255 pixels for width and height
c_dim = 1
input_size = 255

"""Define the model weights and biases
"""
# define the placeholders for inputs and outputs
inputs = tf.placeholder(tf.float32, [None, input_size, input_size, c_dim], name='inputs')

## ------ Add your code here: set the weight of three conv layers
# replace '0' with your hyper parameter numbers 
# conv1 layer with biases: 64 filters with size 9 x 9
# conv2 layer with biases and relu: 32 filters with size 1 x 1
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
weights = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }

biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

path = '/Users/cicho/PycharmProjects/DeepLearning/tf-SRCNN/image/butterfly_GT.bmp'
# the preprocessed image is converted to a an array in float32 data type
image = tf.cast(np.asarray(preprocess(path)), tf.float32)

"""Define the model layers with three convolutional layers
"""
## ------  compute feature maps of input low-resolution images
# conv1 layer with biases and relu : 64 filters with size 9 x 9

conv1 = tf.nn.relu(tf.nn.conv2d(inputs, weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + biases['b1'])
##------compute non-linear mapping
# conv2 layer with biases and relu: 32 filters with size 1 x 1

conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + biases['b2'])
##------ compute the reconstruction of high-resolution image
# conv3 layer with biases and NO relu: 1 filter with size 5 x 5
conv3 = tf.nn.conv2d(conv2, weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + biases['b3']


"""Load the pre-trained model file
"""
model_path = './model/model.npy'
model = np.load(model_path, encoding='latin1').item()

##------ show the weights of model to visualise

weights_w1 = model['w1']
weights_w2 = model['w2']
weights_w3 = model['w3']

biases_b1 = model['b1']
biases_b2 = model['b2']
biases_b3 = model['b3']

# 1st Convolutional layer show 1st filter and 10th filter bias in command window
print('1st Convolutional layer\n1st filter weight value (size 9x9):\n', weights_w1[:, :, 0, 0])
print('bias of 10th filter:\n', biases_b1[9])

# 2nd Convolutional layer show the 5th filter value and 6th filter bias in command window
print('2nd Convolutional layer\n5th filter weight value (size 1x1:', weights_w2[:, :, 0, 4])
print('bias of 6th filter:\n', biases_b2[5])

# 3rd convolutional layer: show the 1st filter value and the 1st filter bias in
print('3rd Convolutional layer\n1st filter weight size(5x5):\n', weights_w3[:, :, 0, 0])
print('bias of 1st  filter:\n', biases_b3[0])

def visualise_w1():
    fig = plt.figure(figsize=(8, 8))

    for i in range(64):
        sub = fig.add_subplot(8, 8, i + 1)
        sub.imshow(weights_w1[:, :, 0, i], cmap='gray')
        plt.axis('off')

    plt.show()


def visualise_w2():
    fig = plt.figure(figsize=(8, 8))

    for i in range(32):
        sub = fig.add_subplot(4, 8, i + 1)
        sub.imshow(weights_w2[:, :, 0, i] + 1, cmap='gray')
        plt.axis('off')
    plt.show()

def visualise_w3():
    fig = plt.figure(figsize=(2, 2))
    plt.imshow(weights_w3[:, :, 0, 0], cmap='gray')
    plt.show()



# visualise weights for layer 1, 2 and 3
# visualise_w1()
# visualise_w2()
# visualise_w3()

"""
save the weights for each layer
"""
# save the 64th weight of layer 1
#scipy.misc.imsave('w1_64.jpg', weights_w1[:, :, 0, 63])
# save the 30th weight of layer 2
#scipy.misc.imsave('W2_30.jpg', weights_w2[0, 0, 0, 29])
# save the weight of layer 3
#scipy.misc.imsave('W3.jpg', weights_w3[0, 0, 0, 0])


"""Initialize the model variables (w1, w2, w3, b1, b2, b3) with the pre-trained model file
"""
# launch a session
sess = tf.Session()

for key in weights.keys():
  sess.run(weights[key].assign(model[key]))

for key in biases.keys():
  sess.run(biases[key].assign(model[key]))

"""Read the test image
"""
blurred_image, groundtruth_image = preprocess('./image/butterfly_GT.bmp')

"""Run the model and get the SR image
"""
# transform the input to 4-D tensor
input_ = np.expand_dims(np.expand_dims(blurred_image, axis=0), axis=-1)

# run the session
# get the feature maps for conv1 and conv2 and output layer
output_ = sess.run(conv3, feed_dict={inputs: input_})
output_2 = sess.run(conv2, feed_dict={inputs: input_})
output_1 = sess.run(conv1, feed_dict={inputs: input_})

# convert output to data type float 64bit
out = output_.astype(np.float64)

##------  Visualize the feature maps for each hidden convolutional layer
def visualise_conv1():
    fig = plt.figure(figsize=(8, 8))

    for i in range(64):
        sub = fig.add_subplot(8, 8, i + 1)
        sub.imshow(output_1[0, :, :, i], cmap='gray')
        plt.axis('off')

    plt.show()


def visualise_conv2():
    fig = plt.figure(figsize=(8, 8))

    for i in range(32):
        sub = fig.add_subplot(4, 8, i + 1)
        sub.imshow(output_2[0, :, :, i], cmap='gray')
        plt.axis('off')

    plt.show()


# visualise_conv1()
# visualise_conv2()
##--------------- save ground truth and blurred images

#groundtruth = scipy.misc.imsave('gt.jpg', groundtruth_image)
#blurred = scipy.misc.imsave('blurred_image.jpg', blurred_image)

# The output must be reduced to 2 Dimensional to have a size of 243x243
sr_image = out.squeeze()
print('size of super-resolution image: ', sr_image.shape)

# save super-resolution image
#sr = scipy.misc.imsave('sr_image.jpg', sr_image)
fig = plt.figure(figsize=(8, 8))
for i in range(64):
    sub = fig.add_subplot(8, 8, i + 1)
    sub.imshow(model['w1'][:, :, 0, i])
    plt.axis('off')
plt.show()

"""compute the PSNR for ground truth image with bicubic interpolation and groundtruth with super-resolution image
"""
# the size of the resulting super-resolution image (243x243) is not consistent with the size of the groundtruth image (255x255)
# crop 6 pixels from each border in width and height
print('ground truth image size: ', groundtruth_image.shape)
cropped_gt = groundtruth_image[6:249, 6:249]
print('cropped GT size: ', cropped_gt.shape)

# compare the PSNR value for each image
psnr_bic = skimage.measure.compare_psnr(groundtruth_image, blurred_image)
psnr_sr = skimage.measure.compare_psnr(cropped_gt, sr_image)

print('Super Resolution: ', psnr_sr, 'dB')
print('Bicubic interpolation: ', psnr_bic, 'dB')