"""
Extract Deep Print Model's fingeprint features
"""

import tensorflow as tf
import numpy as np
from imageprocessing import preprocess_test

## preprocess outline

# R = '/content/drive/MyDrive/Inception_mdl/dataset'
# I = preprocess_test([R + '/101_1.tif', R + '/101_2.tif', R + '/102_1.tif', R + '/102_4.tif'], config)
# print(I.shape)

## code to initialize .pb file uses tf-1.15

# with tf.gfile.GFile("../mdl.pb", "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

# with tf.Graph().as_default() as graph:
#     tf.import_graph_def(graph_def, name="")

# with tf.Session(graph=graph) as sess:
#     # Get a reference to the input tensor
#     inputs = graph.get_tensor_by_name("inputs:0")
#     phase = graph.get_tensor_by_name("phase_train:0")
#     keepProb = graph.get_tensor_by_name("keep_prob:0")
    

#     # Get a reference to the output tensor
#     output_tensor = graph.get_tensor_by_name("outputs:0")
#     moutputs = graph.get_tensor_by_name("MinutiaeFeatures:0")

# # Run the session and get the output
# A, B = sess.run([output_tensor, moutputs], feed_dict={inputs: I, phase : False, keepProb : 1.0})
# F = np.hstack([A, B])
# norm = np.linalg.norm(F, axis=1)
# F_norm = F / norm[:, np.newaxis]
# print(F_norm[0, :] @ F_norm[1 , :])
# print(F_norm[0, :] @ F_norm[2 , :])
# print(F_norm[0, :] @ F_norm[3 , :])
# print(F_norm[2, :] @ F_norm[3 , :])
