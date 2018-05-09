import _pickle as pkl

class reader:
	def __init__(self, params):
		self.__train_file = open(params['train_dir'], 'rb')
		self.__valid_file = open(params['valid_dir'], 'rb')
		self.__test_file = open(params['test_dir'], 'rb')
		self.__train_epoch = 0
		self.__valid_epoch = 0
		self.__test_epoch = 0
		self.__n_epochs = params['n_epochs']

	def train_read(self):
		try:
			yield pkl.load(self.__train_file)
		except:
			self.__train_epoch+=1
			self.__train_file.seek(0)
			yield pkl.load(self.__train_file)


	def valid_read(self):
		try:
			yield pkl.load(self.__valid_file)
		except:
			self.__valid_epoch+=1
			self.__valid_file.seek(0)
			yield pkl.load(self.__valid_file)

	def test_read(self):
		try:
			yield pkl.load(self.__test_file)
		except:
			self.__test_epoch+=1
			self.__test_file.seek(0)
			yield pkl.load(self.__test_file)

	def get_train_epoch(self):
		return self.__train_epoch

	def get_valid_epoch(self):
		return self.__valid_epoch

	def get_test_epoch(self):
		return self.__test_epoch