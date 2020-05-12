import tensorflow as tf
import numpy as np
import glob
import os

checkpoint_dir = './1900_evotuned_weights_random_init_ckpt/model-13120' 
target_dir = './1900_evotuned_weights_random_init' 


def dump_weights(sess, dir_name):
    """
    Saves the weights of the model in dir_name in the format required 
    for loading in this module. Must be called within a tf.Session
    For which the weights are already initialized.
    """
    vs = tf.trainable_variables()
    for v in vs:
        name = v.name
        value = sess.run(v)
        print(name)
        np.save(os.path.join(dir_name,name.replace('/', '_') + ".npy"), np.array(value))


with tf.Session() as sess:
    # Restore variables from disk.
    saver = tf.train.import_meta_graph(checkpoint_dir + '.meta')
    saver.restore(sess, checkpoint_dir)
    print("Variables restored from %s, writing to target dir %s." % (checkpoint_dir, target_dir))
    print("Saved variables:")
    dump_weights(sess, dir_name=target_dir)
