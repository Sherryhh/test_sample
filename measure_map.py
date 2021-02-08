import os
import numpy as np
import pprint
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
import keras
from skimage.transform import resize
from collections import defaultdict
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tfconfig = tf.ConfigProto() 
tfconfig.gpu_options.allow_growth=True 
#tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3 
session = tf.Session(config=tfconfig) 
tf.keras.backend.set_session(session)
tf.set_random_seed(1234)
np.random.seed(0)

# get predicated map and bbox
def get_map(pred, gt, f):
	T = {}
	P = {}
	fx1, fx2, fx3 = f

	for bbox in gt:
		bbox['bbox_matched'] = False

	pred_probs = np.array([s['prob'] for s in pred])
	box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

	for box_idx in box_idx_sorted_by_prob:
		pred_box = pred[box_idx]
		pred_class = pred_box['class']
		pred_x1 = pred_box['x1']
		pred_x2 = pred_box['x2']
		pred_x3 = pred_box['x3']
		pred_r = pred_box['r']
		pred_prob = pred_box['prob']
		if pred_class not in P:
			P[pred_class] = []
			T[pred_class] = []
		P[pred_class].append(pred_prob)
		found_match = False
		for gt_box in gt:
			gt_class = gt_box['class']
			gt_x1 = gt_box['x1']/fx1
			gt_x2 = gt_box['x2']/fx2
			gt_x3 = gt_box['x3']/fx3
			gt_r = gt_box['r']/fx1
			gt_seen = gt_box['bbox_matched']
			if gt_class != pred_class:
				continue
			if gt_seen:
				continue
			iou = data_generators.iou_r((pred_x1, pred_x2, pred_x3, pred_r), (gt_x1, gt_x2, gt_x3, gt_r))
			if iou >= 0.3: #orginal is 0.5
				found_match = True
				gt_box['bbox_matched'] = True
				break
			else:
				continue

		T[pred_class].append(int(found_match))

	for gt_box in gt:
		if not gt_box['bbox_matched']:
		#if not gt_box['bbox_matched'] and not gt_box['difficult']:
			if gt_box['class'] not in P:
				P[gt_box['class']] = []
				T[gt_box['class']] = []

			T[gt_box['class']].append(1)
			P[gt_box['class']].append(0)

	return T, P

# format test image to agree with train image size
def format_img(img, C):
	img_min_side = float(C.im_size)
	(depth, height, width) = img.shape
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = int(img_min_side)
		resized_dense = int(img_min_side)
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = int(img_min_side)
		resized_dense = int(img_min_side)

	fx = width/float(resized_width)
	fy = height/float(resized_height)
	fz = depth/float(resized_dense)

	img = resize(img, (resized_dense, resized_height, resized_width))

	img = img.astype(np.float32)
	img /= C.img_scaling_factor
	img = np.expand_dims(img, axis=0)
	img = np.expand_dims(img, axis=0)
	if K.image_data_format() == 'channels_last':
		img = np.transpose(img, (0, 2, 3, 4, 1))
	return img, fz, fy, fx

# measurements + process output
def rscore(y_true, y_pred):
    i = set(y_true).intersection(y_pred)
    return len(i) / len(y_true)

def fscore(recall, precision):
    if recall + precision == 0:
        return 0
    else:
        return 2 * (precision * recall) / (precision + recall)

def process(mapall):
	res = defaultdict(int)
	for key, value in mapall.items():
		res[key] = sum(value)/len(value)
	return res

sys.setrecursionlimit(40000)

parser = OptionParser()
#data/label_test-short.txt
parser.add_option("-p", "--path", dest="test_path", help="Path to test data.",default='label_test.txt')
parser.add_option("-n", "--num_rois", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=16)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="Fresnet.pickle")
parser.add_option("-o", "--parser", dest="parser", default="simple")
parser.add_option("-f","--file", dest="model", help="Path to weights", default='Fresnet.hdf5')
parser.add_option("--file_record", dest="rec", help="Path to save 2D results", default='results.txt')

(options, args) = parser.parse_args()

f_rec = open(options.rec,'w')

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')

if options.parser == 'simple':
	from keras_frcnn.simple_parser import get_data
else:
	raise ValueError("Command line option parser must be 'simple'")

config_output_filename = options.config_filename

#C = config.Config()
if os.path.getsize(config_output_filename) == 0:
	raise ValueError("empty config")
with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)
	#pickle.dump(C, f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

channel_size = 128 #default
if C.network == 'resnet3d':
    channel_size = 512
	import keras_frcnn.resnet3d_v2 as nn
elif C.network == 'vgg3d':
    channel_size = 512
	import keras_frcnn.vgg3d_v2 as nn
elif C.network == 'net3d':
    channel_size = 128
	import keras_frcnn.net3d as nn
elif C.network == 'snet':
    channel_size = 13
	from keras_frcnn import snet as nn
else:
	print('Not a valid model')
	raise ValueError

if options.model:
	print("valid model")
	C.model_path = options.model
else:
	# set the path to weights based on backend and model
	raise ValueError("not vaild model")

img_path = options.test_path

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if K.image_data_format() == 'channels_first':
	input_shape_img = (1, None, None, None)
	input_shape_features = (channel_size, None, None, None)
else:
	input_shape_img = (None, None, None, 1)
	input_shape_features = (None, None, None, channel_size)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)


model_classifier = Model([feature_map_input, roi_input], classifier)


try:
	print('loading weights from {}'.format(C.model_path))
	model_rpn.load_weights(C.model_path, by_name=True)
	model_classifier.load_weights(C.model_path, by_name=True)
except:
	raise ValueError('Could not load model weights.')


model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs, _, _ = get_data(options.test_path)
test_imgs = [s for s in all_imgs]

T = {}
P = {}
images_map = []
recall_map = []
f1_map = []
apcls = {}
afcls = {}
arcls = {}

for idx, img_data in enumerate(test_imgs):
	if idx%50 == 0:
		print('{}/{}'.format(idx,len(test_imgs)))
	st = time.time()
	filepath = img_data['filepath']

	img = np.load(filepath)

	X, fx1, fx2, fx3 = format_img(img, C)

	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	#predict returns  Numpy array(s) of predictions.
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=1)
	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0] // C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0] // C.num_rois:
			# pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
		for ii in range(P_cls.shape[1]):
			if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1): # if bg
				continue
			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x1, x2, x3, r) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx1, tx2, tx3, tr) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
				tx1 /= C.classifier_regr_std[0]
				tx2 /= C.classifier_regr_std[1]
				tx3 /= C.classifier_regr_std[2]
				tr /= C.classifier_regr_std[3]
				x1, x2, x3, r = roi_helpers.apply_regr(x1, x2, x3, r, tx1, tx2, tx3, tr)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride * x1, C.rpn_stride * x2, C.rpn_stride * x3, C.rpn_stride * r])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))
	
	all_dets = []
	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.7) #original = 0.7
		for jk in range(new_boxes.shape[0]):
			(x1, x2, x3, r) = new_boxes[jk, :]

			det = {'x1': x1, 'x2': x2, 'x3': x3, 'r': r, 'class': key, 'prob': new_probs[jk]}
			all_dets.append(det)

			f_rec.write('{},{},{},{},{},{},{},{},{},{}\n'.format(filepath, x1, x2, x3, r, key, new_probs[jk], fx1, fx2, fx3))
			f_rec.flush()

	if idx%100 == 0:
		print('Elapsed time = {}'.format(time.time() - st))

	t, p = get_map(all_dets, img_data['bboxes'], (fx1, fx2, fx3))
	for key in t.keys():
		if key not in T:
			T[key] = []
			P[key] = []
		T[key].extend(t[key])
		P[key].extend(p[key])
	
	all_aps = []
	all_recall = []
	all_f1 = []

	for key in T.keys():
		Label = []
		Predict = []
		ap = average_precision_score(T[key], P[key])
		if np.isnan(ap):
			ap = 1 #avoid zero division
		all_aps.append(ap)
		if key not in apcls:
			apcls[key] = []
		else:
			apcls[key].append(ap)
		for i in range(len(P[key])):
			Label.append(key)
			if P[key][i] > 0.3:
				Predict.append(key)
			else:
				Predict.append('bg')
		recall = recall_score(Label, Predict, average='micro', zero_division=1)
		if key not in arcls:
			arcls[key] = []
		else:
			arcls[key].append(recall)
		all_recall.append(recall)

		f1 = fscore(recall, ap)
		if key not in afcls:
			afcls[key] = []
		else:
			afcls[key].append(f1)
		all_f1.append(f1)

	images_map.append(np.mean(np.array(all_aps)))
	recall_map.append(np.mean(np.array(all_recall)))
	f1_map.append(np.mean(np.array(all_f1)))


print('mAP for each class:', process(apcls))
print('All mAP = {}'.format(np.mean(np.array(images_map))))
print('Recall Score for each class:', process(arcls))
print('Recall Score averaged out of all class:', np.mean(np.array(recall_map)))
print('F1 score:', process(afcls))
print('Average F1 score:', np.mean(np.array(f1_map)))
f_rec.close()