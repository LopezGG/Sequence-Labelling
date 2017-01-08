from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge, Permute, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional,Input,merge
from keras.layers.recurrent import LSTM, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import accuracy
from keras.utils import np_utils
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop,SGD
from keras.layers import ChainCRF
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D




def build_model(parameters,embedding_matrix =None, weightsPath = None):
	lstm_dim = parameters['word_lstm_dim']
	word_vocab_size = parameters['word_vocab_size'] 
	char_vocab_size = parameters['char_vocab_size']
	char_embedding_dim = parameters['char_dim']
	word_embedding_dim = parameters['word_dim']
	maxCharSize = parameters['maxCharSize']	
	cap_size = 	parameters['cap_size']
	cap_embed_size = parameters['cap_dim']
	max_words = parameters['max_words']
	nb_filters = parameters['cnn_nb_filters']
	window_length = parameters['cnn_window_length']
	learning_rate = parameters['learning_rate']
	decay_rate = parameters['decay_rate'] 
	momentum = parameters['momentum']
	clipvalue = parameters['clipvalue']
	tag_label_size = parameters['tag_label_size']
	dropout = parameters['dropout']

	char_input = Input(shape=(maxCharSize * max_words,), dtype='int32', name='char_input')
	char_emb = Embedding(char_vocab_size, char_embedding_dim, input_length=max_words*maxCharSize, dropout=dropout, name='char_emb')(char_input)
	char_cnn = Convolution1D(nb_filter=nb_filters,filter_length= window_length, activation='tanh', border_mode='full') (char_emb) 
	char_max_pooling = MaxPooling1D(pool_length=maxCharSize) (char_cnn) #  get output per word. this is the size of the hidden layer

	"""

	Summary for char layer alone.
	____________________________________________________________________________________________________
	Layer (type)                     Output Shape          Param #     Connected to 
	====================================================================================================
	char_input (InputLayer)          (None, 2000)          0           None refers to batch size             
	____________________________________________________________________________________________________
	char_emb (Embedding)             (None, 2000, 25)      1250        char_input[0][0] 
	____________________________________________________________________________________________25 is embedding dimension
	convolution1d_1 (Convolution1D)  (None, 2002, 30)      2280        char_emb[0][0]
	____________________________________________________________________________________________30 is the number of filters plus 2 because we use full padding
	maxpooling1d_1 (MaxPooling1D)    (None, 100, 30)        0           convolution1d_1[0][0]
	=============================================================================================max poolign to get 100 hidden units which will be carried over 
	Total params: 3530

	"""


	#based on https://github.com/pressrelations/keras/blob/a2d358e17ea7979983c3c6704390fe2d4b29bbbf/examples/conll2000_bi_lstm_crf.py
	word_input = Input(shape=(max_words,), dtype='int32', name='word_input')
	if (embedding_matrix is not None):
		word_emb = Embedding(word_vocab_size+1, word_embedding_dim,weights=[embedding_matrix], input_length=max_words, dropout=0, name='word_emb')(word_input)
	else:
		word_emb = Embedding(word_vocab_size+1, word_embedding_dim, input_length=max_words, dropout=0, name='word_emb')(word_input)

	caps_input = Input(shape=(max_words,), dtype='int32', name='caps_input')
	caps_emb = Embedding(cap_size, cap_embed_size, input_length=None, dropout=dropout, name='caps_emb')(caps_input)
	#concat axis refers to the axis whose dimension can be different
	total_emb = merge([word_emb, caps_emb,char_max_pooling], mode='concat', concat_axis=2,name ='total_emb')
	emb_droput = Dropout(dropout)(total_emb)
	#inner_init : initialization function of the inner cells. I believe this is Cell state
	bilstm_word  = Bidirectional(LSTM(lstm_dim,inner_init='uniform', forget_bias_init='one',return_sequences=True))(emb_droput)
	bilstm_word_d = Dropout(dropout)(bilstm_word)

	dense = TimeDistributed(Dense(tag_label_size))(bilstm_word_d)
	crf = ChainCRF()
	crf_output = crf(dense)
	#to accoutn for gradient clipping
	#info on nesterov http://stats.stackexchange.com/questions/211334/keras-how-does-sgd-learning-rate-decay-work
	sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum, nesterov=False,clipvalue = clipvalue)



	model = Model(input=[word_input,caps_input,char_input], output=[crf_output])
	if(weightsPath):
		model.load_weights(weightsPath)
	model.compile(loss=crf.sparse_loss,
	              optimizer=sgd,
	              metrics=['sparse_categorical_accuracy'])

	model.summary()
	return model

def train_model (model,parameters,Words_id_train,caps_train,char_train,tag_train,Words_id_dev=None,caps_dev=None,char_dev = None,tag_dev=None):
	
	# define the checkpoint
	filepath="weights-improvement-BiLSTM-All-no-wd-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	batch_size = parameters['batch_size']
	epoch_number = parameters['epoch_number']
	model.fit([Words_id_train,caps_train,char_train], tag_train,
          batch_size=batch_size,
          validation_data=([Words_id_dev,caps_dev,char_dev], tag_dev), nb_epoch=epoch_number,callbacks=callbacks_list)
	return model