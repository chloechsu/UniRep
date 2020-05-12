import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from unirep import babbler1900, babbler64


def nonpad_len(batch):
    nonzero = batch > 0
    lengths = np.sum(nonzero, axis=1)
    return lengths    


def load_sarkisyan(seqs_filename, brightness_filename, data_filename='data/sarkisyan.csv'):
    # Load sarkisyan dataset
    if os.path.exists(seqs_filename) and os.path.exists(brightness_filename):
        seqs = np.loadtxt(seqs_filename)
        brightness = np.loadtxt(brightness_filename)
    else:
        sarkisyan = pd.read_csv(data_filename)
        sequences = []
        brightness = []
        stop_codon_cnt = 0
        for i, row in sarkisyan.iterrows():
            seq = row.seq.strip('*')
            if b.is_valid_seq(seq) and len(seq) < 275:
                sequences.append(b.format_seq(seq))
                brightness.append(row.medianBrightness)
            else:
                if '*' in seq:
                    stop_codon_cnt += 1
                else:
                    print('Invalid seq', seq)
        seqs = np.stack(sequences)
        brightness = np.array(brightness)[:, None]
        print('Formatted %d sequences. Discarded %d with stop codon.' % (seqs.shape[0], stop_codon_cnt))
        np.savetxt(seqs_filename, seqs)
        np.savetxt(brightness_filename, brightness)
    return seqs, brightness


def get_final_hidden_vals(final_hidden_vals_filename, seqs, model_weight_path,
        batch_size=256, use_checkpoint=False):
    if os.path.exists(final_hidden_vals_filename):
        final_hidden_vals = np.loadtxt(final_hidden_vals_filename)
    else:
        babbler_class = babbler1900
        if '64' in model_weight_path:
            babbler_class = babbler64
        b = babbler_class(batch_size=batch_size, model_path=model_weight_path,
                load_checkpoint=True)
        if use_checkpoint:
            saver = tf.train.Saver()
        final_hidden_op, x_placeholder, batch_size_placeholder, seq_length_placeholder, initial_state_placeholder = (
            b.get_rep_ops())
        final_hidden_vals = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if use_checkpoint:
                saver.restore(sess, model_weight_path)
            n_batches = int(seqs.shape[0] / batch_size)
            leftover = seqs.shape[0] % batch_size
            n_batches += int(bool(leftover))
            for i in range(n_batches):
                if i == n_batches - 1:
                    batch = seqs[-batch_size:]
                else:
                    batch = seqs[i*batch_size:(i+1)*batch_size]
                length = nonpad_len(batch)
                final_hidden_ = sess.run(
                    final_hidden_op,
                    feed_dict={
                        x_placeholder: batch,
                        batch_size_placeholder: batch.shape[0],
                        seq_length_placeholder: length,
                        initial_state_placeholder:b._zero_state
                    })
                if i == n_batches - 1:
                    final_hidden_vals.append(final_hidden_[-leftover:])
                else:
                    final_hidden_vals.append(final_hidden_)

        final_hidden_vals = np.concatenate(final_hidden_vals, axis=0)
        np.savetxt(final_hidden_vals_filename, final_hidden_vals)
    print('Ran inference on %d sequences. Saved results to %s.' %
            (seqs.shape[0], final_hidden_vals_filename))
    return final_hidden_vals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_weight_path', type=str)
    parser.add_argument('--seqs_filename', type=str,
            default='data/sarkisyan_filtered_seqs.txt')
    parser.add_argument('--brightness_filename', type=str,
            default='data/sarkisyan_filtered_brightness.txt')
    parser.add_argument('--final_hidden_filename_prefix', type=str,
            default='data/sarkisyan_final_hidden_vals')
    parser.add_argument('--use_checkpoint', action='store_true')
    args = parser.parse_args()

    seqs, brightness = load_sarkisyan(args.seqs_filename, args.brightness_filename)

    final_hidden_vals_filename = 'data/%s_%s.txt' % (
            args.final_hidden_filename_prefix, args.model_weight_path.strip('./'))
    final_hidden_vals = get_final_hidden_vals(final_hidden_vals_filename, seqs,
            args.model_weight_path, args.use_checkpoint)
