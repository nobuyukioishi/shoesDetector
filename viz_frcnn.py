import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import random
from keras.utils import plot_model
import pydot


sys.setrecursionlimit(40000)
		
# parser = OptionParser()

# parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
# parser.add_option("-n", "--num_rois", dest="num_rois",
# 				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
# parser.add_option("--output_config_filename", dest="config_filename", help=
# 				"Location to read the metadata related to the training (generated when training).",
# 				default="config.pickle")
# parser.add_option("--img_output", dest="img_out_path", help="Location to output the tested data images") 

# (options, args) = parser.parse_args()
# print((options, args))
# if not options.test_path:   # if filename is not given
# 	parser.error('Error: path to test data must be specified. Pass --path to command line')


# config_output_filename = options.config_filename
config_output_filename = "model_frcnn_11.pickle"
# config_output_filename = "config.pickle"

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)
# print("debug")
# print(C.im_size)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

# img_path = options.test_path


def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

# print("class_mapping")
# print(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}

colors ={ 	'shoe'	 :	(0, 0,	255), 
			'slipper':	(255,	0,	0),
			'sandal' :	(0	, 255,	0),
			'bg'	 :	(0	,	0,	0)
		}

class_to_color = {class_mapping[v]: colors[class_mapping[v]] for index, v in enumerate(class_mapping)}

# C.num_rois = int(options.num_rois)
C.num_rois = 32

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (1024, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, 1024)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')


# plot_model(model_classifier, to_file='model.png')  # outputed picture
# print("outputed")

# get the symbolic outputs of each "key" layer (we gave them unique names).
model_classifier_layer_dict = dict([(layer.name, layer) for layer in model_classifier.layers])
model_rpn_layer_dict = dict([(layer.name, layer) for layer in model_rpn.layers])


#input img
fileName= "1.png"
#prepare input picture
im = cv2.imread(fileName)
if im is None:
	print("no image file selected")
# im=cv2.resize(im, (56,56))
imNP = np.asarray(im)
a= [imNP.reshape(-1,imNP.shape[0],imNP.shape[1],imNP.shape[2]), 0]

layer_output = K.function([model_rpn.layers[0].input, K.learning_phase()], [model_rpn_layer_dict['add_16']])

layer_output = layer_output(a)[0]

print('The second dimension tells us how many convolutions do we have: %s (%d convolutions)' % (str(layer_output.shape),layer_output.shape[1]))
print(layer_output.shape[3])
numberConvolution=layer_output.shape[3]
b= int(numberConvolution/8)
a= int(numberConvolution/b)

i=1
fig = plt.figure() 
fig.canvas.set_window_title('My Window Title') 

for onePic in np.rollaxis(layer_output, 3):
	onePic = onePic.reshape(layer_output.shape[1],layer_output.shape[2])
	plt.subplot(a,b, i)
	plt.imshow(onePic,'gray')
	plt.xticks([]),plt.yticks([])
	i=i+1

plt.show()

cv2.waitKey(1)
