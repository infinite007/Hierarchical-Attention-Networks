import tensorflow as tf
import numpy as np
import model
from reader import reader

params = {
	"n_sents":100,
	"n_words":100,
	"embedding_size":50
}



params["vocab_size"] = 100+1
params["num_classes"] = 5
params["train_dir"] = './data/train.pkl'
params["valid_dir"] = './data/valid.pkl'
params["test_dir"] = './data/test.pkl'

rdr = reader()
model = model.Model(params)
model.forward()
init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

with tf.Session() as sess:
	sess.run(init_op)
	inp = np.random.random_integers(0,100, (1, params['n_sents'], params['n_words']))
	feed_dict = {model.input:inp}
	states, pred = sess.run([model.final_inputs, model.preds], feed_dict=feed_dict)
	print(states.shape, pred)