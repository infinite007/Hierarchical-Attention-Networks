import  tensorflow as tf

class Model:
	def __init__(self, params):
		self.params = params
		self.input = tf.placeholder(dtype=tf.int32,shape=[None, self.params['sentence_length'], self.params['word_length']])
		self.target = tf.placeholder(dtype=tf.int32,shape=[None])
		self.word_embeddings = tf.get_variable('embeddings',[self.params['vocab_size'], self.params['embedding_size']])

	def forward(self):
		word_inputs_raw = tf.nn.embedding_lookup(self.word_embeddings, self.input)
		word_gru_cells = tf.nn.rnn_cell.GRUCell(self.params['num_units'])
		word_gru_colony = tf.nn.bidirectional_dynamic_rnn(wo)

	def backward(self):
		pass

	def infer(self):
		pass