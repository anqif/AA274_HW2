import argparse
import numpy as np
import tensorflow as tf
from utils import *

GRAPH = 'retrained_graph.pb'
with open('labels.txt', 'r') as f:
    LABELS = f.read().split()

def compute_brute_force_classification(image_path, nH=8, nW=8):
    raw_image = decode_jpeg(image_path)    # H x W x 3 numpy array (3 for each RGB color channel)

    #### EDIT THIS CODE
    # Determine indices of padded window idx
    def window_idx(idx, step, bounds = [0, np.Inf], pad = [0, 0]):
        if step <= 0:
            raise ValueError('step must be a positive integer')
        if bounds[1] < bounds[0]:
            raise ValueError('upper bound must be greater than lower bound')
        if any(p < 0 for p in pad):
            raise ValueError('pad must be an array of non-negative integers')

        start = max(idx*step - pad[0], bounds[0])
        end = min((idx + 1)*step + pad[1], bounds[1])
        return start, end

    nclasses = 3   # (neither, dog, cat)
    rows, cols, chans = raw_image.shape
    rstep = int(np.floor(1.0*rows/nH))   # take floor to stay within image bounds
    cstep = int(np.floor(1.0*cols/nW))
    rpad = 2*[int(np.round(0.15*rows))]   # window padding
    cpad = 2*[int(np.round(0.15*cols))]
    window_predictions = np.empty([nH, nW, nclasses])

    with tf.Session() as sess:
        for r in range(nH):
            for c in range(nW):
                rstart, rend = window_idx(r, rstep, [0, rows], rpad)
                cstart, cend = window_idx(c, cstep, [0, cols], cpad)
                window = raw_image[rstart:rend, cstart:cend]
                window_predictions[r,c,:] = classify_image(window, sess)
    ####

    return np.squeeze(np.array(window_predictions))

def compute_convolutional_KxK_classification(image):
    graph = tf.get_default_graph()
    classification_input_tensor =  graph.get_tensor_by_name('bottleneck_input/BottleneckInputPlaceholder:0')
    classification_output_tensor = graph.get_tensor_by_name('final_result:0')
    convolution_ouput_tensor = graph.get_tensor_by_name('bottleneck_grid:0')    # 'mixed_10/join:0' in Inception-v3
    K = convolution_ouput_tensor.shape[0]

    with tf.Session() as sess:
        convolution_output = run_with_image_input(convolution_ouput_tensor, image, sess)
        predictionsKxK = sess.run(classification_output_tensor,
                                  {classification_input_tensor: np.reshape(convolution_output, [K*K, -1])})
        return np.reshape(predictionsKxK, [K,K,-1])

def compute_and_plot_saliency(image):
    graph = tf.get_default_graph()
    logits_tensor = graph.get_tensor_by_name('final_training_ops/Wx_plus_b/logits:0')

    with tf.Session() as sess:
        logits = np.squeeze(run_with_image_input(logits_tensor, image, sess))
        top_class = np.argmax(logits)
        w_ijc = gradient_of_class_score_with_respect_to_input_image(image, top_class, sess)    # defined in utils.py
        M = tf.reduce_max(tf.abs(w_ijc), axis = [2], keepdims = False).eval()   # compute class saliency map M_ij = max_c |w_ijc|

    plt.subplot(2,1,1)
    plt.imshow(M)
    plt.title('Saliency with respect to predicted class %s' % LABELS[top_class])
    plt.subplot(2,1,2)
    plt.imshow(decode_jpeg(image))
    plt.show()

def plot_classification(image_path, classification_array):
    nH, nW, _ = classification_array.shape
    image_data = decode_jpeg(image_path)
    aspect_ratio = float(image_data.shape[0]) / image_data.shape[1]
    plt.figure(figsize=(8, 8*aspect_ratio))
    p1 = plt.subplot(2,2,1)
    plt.imshow(classification_array[:,:,0], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[0])
    p1.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    p2 = plt.subplot(2,2,2)
    plt.imshow(classification_array[:,:,1], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[1])
    p2.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    p2 = plt.subplot(2,2,3)
    plt.imshow(classification_array[:,:,2], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[2])
    p2.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(image_data)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--scheme', type=str)
    FLAGS, _ = parser.parse_known_args()

    load_graph(GRAPH)
    if FLAGS.scheme == 'brute':
        plot_classification(FLAGS.image, compute_brute_force_classification(FLAGS.image, 8, 8))
    elif FLAGS.scheme == 'conv':
        plot_classification(FLAGS.image, compute_convolutional_KxK_classification(FLAGS.image))
    elif FLAGS.scheme == 'saliency':
        compute_and_plot_saliency(FLAGS.image)
    else:
        print 'Unrecognized scheme:', FLAGS.scheme