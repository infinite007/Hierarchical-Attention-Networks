import tensorflow as tf

class Model:
	def __init__(self, params):
		self.params = params
		self.input = tf.placeholder(dtype=tf.int32)
		self.target = tf.placeholder(dtype=tf.int32,shape=[None])
		self.word_embeddings = tf.get_variable('embeddings',[self.params['vocab_size'], self.params['embedding_size']])

	def att_block(self, inputs, out_size, scope):
		gru_cells_forward = tf.nn.rnn_cell.GRUCell(out_size)
		gru_cells_backward = tf.nn.rnn_cell.GRUCell(out_size)
		gru_outputs, gru_states = tf.nn.bidirectional_dynamic_rnn(
																gru_cells_forward,
																gru_cells_backward,
																inputs,
																dtype=tf.float32,
																scope=scope)
		state_reshaped = tf.reshape(gru_states, [-1, 2*out_size])
		gru_outputs_concat = tf.concat(gru_outputs, 2)
		W = tf.Variable(tf.random_uniform([2*out_size, 2*out_size], -1., 1.))
		b = tf.Variable(tf.random_uniform([2*out_size], -1., 1.))
		attention = tf.tanh(tf.add(tf.matmul(state_reshaped, W), b))
		return gru_outputs_concat, attention

	def prediction_layer(self, input,input_shape, num_classes, scope):
		W = tf.Variable(tf.random_uniform([input_shape, num_classes], -1., 1.), scope)
		b = tf.Variable(tf.random_uniform([num_classes], -1., 1.), scope)
		return tf.add(tf.matmul(input, W), b)


	def forward(self):
		input_shape = tf.shape(self.input)
		n_sents = input_shape[1]
		n_words = input_shape[2]
		word_inputs_raw = tf.nn.embedding_lookup(self.word_embeddings, self.input)
		word_inputs_reshaped = tf.reshape(word_inputs_raw, [-1, n_words, self.params['embedding_size']])
		word_outputs, word_attention= self.att_block(word_inputs_reshaped, self.params['embedding_size'], 'word_gru')
		sentence_inputs = tf.multiply(word_outputs, word_attention)
		sentence_inputs_reshaped = tf.reshape(sentence_inputs, [-1,n_sents, 2*self.params['embedding_size']])
		sentence_outputs, sentence_attention = self.att_block(sentence_inputs_reshaped, 2*self.params['embedding_size'], 'sentence_gru')
		final_inputs = tf.multiply(sentence_outputs, sentence_attention)
		#todo error in the line below. resolve the issue.
		self.logits = self.prediction_layer(final_inputs, 4*self.params['embedding_size'], self.params['num_classes'], 'prediction')

	def backward(self):
		pass