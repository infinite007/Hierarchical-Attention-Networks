import  tensorflow as tf

class Model:
	def __init__(self, params):
		self.params = params
		self.input = tf.placeholder(dtype=tf.int32)
		self.target = tf.placeholder(dtype=tf.int32,shape=[None])
		self.word_embeddings = tf.get_variable('embeddings',[self.params['vocab_size'], self.params['embedding_size']])

	def forward(self):
		word_inputs_raw = tf.nn.embedding_lookup(self.word_embeddings, self.input)
		word_gru_cells_forward = tf.nn.rnn_cell.GRUCell(self.params['embedding_size'])
		word_gru_cells_backward = tf.nn.rnn_cell.GRUCell(self.params['embedding_size'])
		word_inputs_reshaped = tf.reshape(word_inputs_raw, [-1, self.params['embedding_size']])
		_, self.gru_output_states = tf.nn.bidirectional_dynamic_rnn(word_gru_cells_forward, word_gru_cells_backward, word_inputs_reshaped, dtype=tf.float32, scope='word_gru')
		# word_W = tf.Variable(tf.random_uniform([2*self.params['embedding_size'], 2*self.params['embedding_size']], -1., 1.))
		# word_b = tf.Variable(tf.random_uniform([2*self.params['embedding_size']], -1., 1.))
		# self.word_att = tf.tanh(tf.add(tf.matmul(word_W, gru_output_states), word_b))

	def backward(self):
		pass