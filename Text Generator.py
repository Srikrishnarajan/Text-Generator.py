#Importing the Dependencies
import numpy
import sys
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

#Loading the Data
file = open('frankenstein-2.txt', encoding = 'utf8').read()

#Tokenization
#Standardization
def tokenize_words(input):
    input = input.lower()
    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)
    
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

processed_inputs = tokenize_words(file)

#Characters to Numbers
chars = sorted(list(set(processed_inputs)))
chars_to_num = dict((c, i) for i, c in enumerate(chars))

#Check if words to characters or characters to numbers has worked.
input_length = len(processed_inputs)
chars_length = len(chars)

print("Total number of characters: ", input_length)
print("Total characters: ", chars_length)

#Sequence Length
seq_length = 100
x_data = []
y_data = []

#Looping through the sequence
for i in range(0, input_length - seq_length, 1):
    in_seq = processed_inputs[i:i + seq_length]
    out_seq = processed_inputs[i + seq_length]
    
    x_data.append([chars_to_num[chars] for chars in in_seq])
    y_data.append(chars_to_num[out_seq])

n_patterns = len(x_data)

print("Total number of patterns: ", n_patterns)

#Convert input sequence to numpy array that our network can use
X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(chars_length)

#One-Hot Encoding
y = np_utils.to_categorical(y_data)

#Creating the Model
model = Sequential()

model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(256, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(128))
model.add(Dropout(0.2))

model.add(Dense(y.shape[1], activation = 'softmax'))

#Compiling the Model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

#Saving Weights
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
desired_callbacks = [checkpoint]

#Fit model and let it train
model.fit(X, y, epochs = 4, batch_size = 256, callbacks = desired_callbacks)

#Recompile the Model with saved weights
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

#Output of the Model back into characters
num_to_chars = dict((i, c) for i, c in enumerate(chars))

#Random seed to help generate
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]

print("Random Seed:")
print("\"", ''.join([num_to_chars[value] for value in pattern]), "\"")

#Generating the Text
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x/float(chars_length)
    
    prediction = model.predict(x, verbose = 0)
    index = numpy.argmax(prediction)
    
    result = num_to_chars[index]
    seq_in = [num_to_chars[value] for value in pattern]
    
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
