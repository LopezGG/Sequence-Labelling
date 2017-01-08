import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
import loader

from utils import models_path, evaluate, eval_script, eval_temp,to_onehot
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from loader import CreateX_Y
from model import build_model,train_model

from keras.models import Sequential
from keras.models import model_from_json
import numpy
import theano, keras
import pickle

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-e", "--epoch", default="50",
    type='int', help="Number of epochs to run"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="30",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="200",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="1",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="4",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-g", "--pre_weights", default="",
    help="path to Pretrained for from a previous run"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Path to Reload the last saved model"
)
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method
parameters['epoch_number'] = opts.epoch 
parameters['weights'] = opts.pre_weights 
# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)


# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']
weights = parameters['weights']


train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)


# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_Words_id_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word,embedding_matrix = augment_with_pretrained(
        dico_Words_id_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None,
        parameters['word_dim']
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_Words_id_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags 
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)




# Index data
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

parameters['word_vocab_size'] = len(dico_words.keys())
parameters['char_vocab_size'] = len(dico_chars.keys())
parameters['cap_size'] = 4 
parameters['batch_size']  = 10 
parameters['cnn_nb_filters']  = 30 
parameters['cnn_window_length'] = 3
#SGD parameters
parameters['learning_rate'] = 0.015
parameters['decay_rate'] = 0.05
parameters['momentum'] = 0.9
parameters['clipvalue'] = 5.0
parameters['max_words'] = 100 #a sentence can have atmost 100 words
parameters['maxCharSize'] = 20 #a word can ahve atmost 20 char


max_words = parameters['max_words']
maxCharSize = parameters['maxCharSize']
Words_id_train,tag_train,caps_train,char_train,Words_str_train = CreateX_Y (train_data,max_words,maxCharSize)
#tag_test_bare for visually checking the actual results
Words_id_test,tag_test_bare,caps_test,char_test,Words_str_test= CreateX_Y (test_data,max_words,maxCharSize)
Words_id_dev,tag_dev,caps_dev,char_dev,Words_str_dev = CreateX_Y (dev_data,max_words,maxCharSize)
parameters['tag_label_size'] = len(tag_to_id.keys())

tag_train = np.expand_dims(tag_train, -1)
tag_dev = np.expand_dims(tag_dev, -1)
tag_test = np.expand_dims(tag_test_bare, -1)

print('X_Words_id_train shape:', Words_id_train.shape)
print('tag_train shape:', tag_train.shape)
print('caps_train shape:', caps_train.shape)
print('char_train shape:', char_train.shape)
print()
print('X_Words_id_dev shape:', Words_id_dev.shape)
print('tag_dev shape:', tag_dev.shape)
print('caps_dev shape:', caps_dev.shape)

print()

print('X_Words_id_test shape:', Words_id_test.shape)
print('tag_test shape:', tag_test.shape)
print('caps_test shape:', caps_test.shape)
print('Words_str_test shape:', len(Words_str_test))
if (parameters['pre_emb'] and weights):
	assert os.path.isfile(weights)
	model = build_model(parameters,embedding_matrix=embedding_matrix,weightsPath=weights)
elif (parameters['pre_emb'] ):
	model = build_model(parameters,embedding_matrix=embedding_matrix)
elif (weights):
	assert os.path.isfile(weights)
	model = build_model(parameters,weightsPath =weights )
else:
    model = build_model(parameters)
print('Train...')
pickle.dump(word_to_id, open("word_to_id.pkl",'wb'))
pickle.dump(char_to_id, open("char_to_id.pkl",'wb'))
pickle.dump(tag_to_id, open("tag_to_id.pkl",'wb'))
pickle.dump(id_to_tag,open("id_to_tag.pkl",'wb'))
pickle.dump(parameters,open("parameters.pkl",'wb'))
model = train_model (model,parameters,Words_id_train,caps_train,char_train,tag_train,Words_id_dev,caps_dev,char_dev,tag_dev)

# summarize performance of the model
scores = model.evaluate([Words_id_test, caps_test,char_test], tag_test, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

#save these for later predictions
pickle.dump(word_to_id, open("word_to_id.pkl",'wb'))
pickle.dump(char_to_id, open("char_to_id.pkl",'wb'))
pickle.dump(tag_to_id, open("tag_to_id.pkl",'wb'))
pickle.dump(id_to_tag,open("id_to_tag.pkl",'wb'))
pickle.dump(parameters,open("parameters.pkl",'wb'))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


predict = model.predict([Words_id_test, caps_test,char_test], verbose=1)
print("predict shape ", predict.shape)
tagset = [] 
for row in predict:
    tag = []
    for prediction in row:
        index = numpy.argmax(prediction)
        tag.append(id_to_tag[index])
    tagset.append(tag)

wordset = []

for row in Words_str_test:
    word = []
    for word_result in row:
        word.append(word_result)
    wordset.append(word)
print("Words_str_test length " , len(wordset))  
    #print(word)
"""
actTagset = []
for row in tag_test_bare:
    actTag = []
    for tag_id in row:
        actTag.append(id_to_tag[tag_id])
    actTagset.append(actTag)



with (open(outFile,'w')) as f:
    for words,tags,actTags in zip(wordset,tagset,actTagset):
        firstword = False
        for word,tag,actTag in zip(words,tags,actTags):
            if((word == "<UNK>") and not firstword ):
                continue
            firstword =True
            f.write(word + "_"+ actTag +"_"+ tag + " ")
        f.write("\n\n");
"""
outFile = "resultsAll.txt"
with (open(outFile,'w')) as f:
    for words,tags in zip(wordset,tagset):       
        firstword = False
        for word,tag in zip(words,tags):
            if((word == "<UNK>") and not firstword ):
                continue
            firstword =True
            try:
                f.write(unicode(word) + "__"+ tag + " ")
            except :
                pass
        f.write("\n\n");
