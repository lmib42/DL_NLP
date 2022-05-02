#! usr/vin/env python3
# coding: 'utf-8'

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import time

from keras.models import load_model

model = load_model('model1.h5')



f = open('.txt', 'r')
box = f.read()
f.close()
text = box

#インデックスを出すための処理
ind_box = [0]
cnt = 0
for i in text:
    cnt+=1
    if i=='\n':
        ind_box.append(cnt)
    else:
        pass

ind_box = ind_box[0:-1]


chars = sorted(list(set(text)))
#print(chars)
#print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 3
step = 1
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

#print('nb sequences:', len(sentences))
#print('Vectorization...')

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

'''
print('Build model...')

model = Sequential()
model.add(LSTM(784, input_shape=(maxlen, len(chars))))
model.add(Activation('relu'))
model.add(Activation('relu'))
model.add(Dense(len(chars)))
model.add(Activation('relu'))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
'''


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#icnt = 0

for iteration in range(300):
    print('')
    inw = input('>>')
    if len(inw)==1:
        break

    #if iteration > 200:
    #    input('>>')
    print('')
    #print('-' * 50)
    #print('Iteration', iteration)
    #model.fit(x, y,
    #          batch_size=128,
    #          epochs=1)

    #model.save('model1.h5')

    #start_index = random.randint(0, len(text) - maxlen - 1)
    
    #icnt +=1

    #if icnt==1:
    random.shuffle(ind_box)
    start_index = ind_box[0]

    for diversity in [0.5]:
        #print()
        #print('----- diversity:', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        #print('----- Generating with seed: "' + sentence + '"')
        #print('')
        sys.stdout.write(generated)

        for i in range(35):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.
    
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            if next_char == '\n':
                break
            else:
   
                generated += next_char
                sentence = sentence[1:] + next_char
                sys.stdout.write(next_char)
                time.sleep(random.randint(40, 100)/ 10000)
                sys.stdout.flush()
        
        print()
        print('')

    #if icnt==24:
        #icnt=0

