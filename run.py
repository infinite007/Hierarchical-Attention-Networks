import tensorflow as tf
import numpy as np
import model

params = {
	"sentence_length":100,
	"word_length":100,
	"embedding_size":50
}

params["vocab_size"] = 100
params["num_classes"] = 5

model = model.Model(params)
model.forward()
init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:
	sess.run(init_op)
	inp = np.random.random_integers(0,100, (1, params['sentence_length'], params['word_length']))
	feed_dict = {model.input:inp}
	states = sess.run(model.logits, feed_dict=feed_dict)
	print(states)