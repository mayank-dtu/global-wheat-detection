import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import matplotlib
import glob
matplotlib.use('TkAgg')
# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# Name of the directory containing the object detection module we're using
#MODEL_NAME = 'inference_graph'
#IMAGE_NAME = 'semaforo.png'
# Grab path to current working directory
#CWD_PATH = os.getcwd()
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
tr=0.5
a=r'/home/user/Mayank/backup/faster_rcnn_inception_v2_coco/Adam/train_1_bs64_lr0002/train_transfer_1/model'
PATH_TO_CKPT = os.path.join(a,'frozen_inference_graph.pb')
#PATH_TO_CKPT = r'/home/user/Mayank/backup/faster_rcnn_inception_v2_coco/Adam/train_1_bs64/model/saved_model/saved_model.pb'
# Path to label map file
PATH_TO_LABELS = r'/home/user/Mayank/global-wheat-detection/label_map.pbtxt'
# Path to image
#PATH_TO_IMAGE = r'/home/user/Mayank/global-wheat-detection/data/test'
# Number of classes the object detector can identify
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
sess = tf.compat.v1.Session(graph=detection_graph)
# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

for fname in os.listdir('/home/user/Mayank/global-wheat-detection/data/test_tr')[:]:
    #Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    PATH_TO_IMAGE = os.path.join('/home/user/Mayank/global-wheat-detection/data/test_tr',fname)
    print(PATH_TO_IMAGE,"########################")
    image = cv2.imread(PATH_TO_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    # Draw the results of the detection (aka 'visulaize the results')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=tr,
        groundtruth_box_visualization_color='black',
        max_boxes_to_draw=None)
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image,
#         np.squeeze(boxes),
#         np.squeeze(classes).astype(np.int32),
#         None,
#         category_index,
#         use_normalized_coordinates=True,
#         line_thickness=4,
#         min_score_thresh=0.5,
#         groundtruth_box_visualization_color='black',
#         max_boxes_to_draw=None)
    print(boxes, scores, classes, num)
    #matplotlib.pyplot.figure(figsize=(20,10))
#     cv2.namedWindow("JK", cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty("JK",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     cv2.imshow("JK",image)
#     cv2.waitKey(0)
    cv2.imwrite(os.path.join(a,"thresh"+str(tr)+fname),image)
#     image = cv2.imread(PATH_TO_IMAGE)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #matplotlib.pyplot.figure(figsize=(20,10))
#     cv2.imshow("JK",image)
#     cv2.waitKey(0)