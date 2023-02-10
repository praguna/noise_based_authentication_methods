"""
Extract Deep Print Model's fingeprint features
"""

import tensorflow as tf
import numpy as np
from imageprocessing import preprocess_test
import sys
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def init_fp_session():
    ## code to initialize .pb file uses tf-1.15
    with tf.gfile.GFile("../dumps/mdl.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        
    sess = tf.Session(graph=graph)
    # Get a reference to the input tensor
    inputs = graph.get_tensor_by_name("inputs:0")
    phase = graph.get_tensor_by_name("phase_train:0")
    keepProb = graph.get_tensor_by_name("keep_prob:0")

    def extract_embeddings(I):    
        # Get a reference to the output tensor
        output_tensor = graph.get_tensor_by_name("outputs:0")
        moutputs = graph.get_tensor_by_name("MinutiaeFeatures:0")

        A, B = sess.run([output_tensor, moutputs], feed_dict={inputs: I, phase : False, keepProb : 1.0})
        F = np.hstack([A, B])
        norm = np.linalg.norm(F, axis=1)
        F_norm = F / norm[:, np.newaxis]
        return F_norm

    return extract_embeddings

if __name__ == "__main__":
    R = "../dumps/dataset"
    P = [R + '/101_1.tif', R + '/101_4.tif' , R + '/102_3.tif' , R + '/102_4.tif'] 
    I = preprocess_test(P)
    extract_embeddings = init_fp_session()
    F_norm = extract_embeddings(I)
    print(F_norm[0, :] @ F_norm[1 , :], "same class")
    print(F_norm[0, :] @ F_norm[2 , :], "same class")
    print(F_norm[0, :] @ F_norm[3 , :], "diff class")
    print(F_norm[2, :] @ F_norm[3 , :], "same class")

