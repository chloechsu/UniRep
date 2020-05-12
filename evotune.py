import tensorflow as tf
import numpy as np

# Set seeds
tf.set_random_seed(42)
np.random.seed(42)

USE_FULL_1900_DIM_MODEL = False # if True use 1900 dimensional model, else use 64 dimensional one.

if USE_FULL_1900_DIM_MODEL:
    # Sync relevant weight files
    # !aws s3 sync --no-sign-request --quiet s3://unirep-public/1900_weights/ 1900_weights/
    # Import the mLSTM babbler model
    from unirep import babbler1900 as babbler
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = "./1900_weights"
    
else:
    # Sync relevant weight files
    # !aws s3 sync --no-sign-request --quiet s3://unirep-public/64_weights/ 64_weights/
    # Import the mLSTM babbler model
    from unirep import babbler64 as babbler
    # Where model weights are stored.
    MODEL_WEIGHT_PATH = "./64_weights"

batch_size = 256
b = babbler(batch_size=batch_size, model_path=MODEL_WEIGHT_PATH)

# Load jackhmmer evotune seqs.
seqlens = []
with open("evotune_seqs/wide_jackhmmer_3_train.txt", "r") as source:
    with open("formatted_evotune_jackhmmer_train.txt", "w") as destination:
        for i,seq in enumerate(source):
            seq = seq.strip()
            if b.is_valid_seq(seq):
                seqlens.append(len(seq))
                formatted = ",".join(map(str,b.format_seq(seq)))
                destination.write(formatted)
                destination.write('\n')
print('Formatted %d sequences.' % len(seqlens))

bucket_op = b.bucket_batch_pad("formatted_evotune_jackhmmer_train.txt", lower=100, upper=500, interval=50)

logits, seqloss, x_placeholder, y_placeholder, batch_size_placeholder, initial_state_placeholder = (
    b.get_babbler_ops())
learning_rate=.001
optimizer = tf.train.AdamOptimizer(learning_rate)
tuning_op = optimizer.minimize(seqloss)
num_iters = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_iters):
        batch = sess.run(bucket_op)
        loss_, __, = sess.run([seqloss, tuning_op],
                feed_dict={
                     x_placeholder: batch,
                     y_placeholder: batch,
                     batch_size_placeholder: batch_size,
                     initial_state_placeholder:b._zero_state
                }
        )
        print("Iteration {0}: {1}".format(i, loss_))
    b.dump_weights(sess, dir_name="./64_evotuned_weights")
