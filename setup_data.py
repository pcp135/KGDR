import numpy as np
import cPickle as pickle

train = np.genfromtxt('train.csv', delimiter=',')
test = np.genfromtxt('test.csv', delimiter=',')
target = train[:,0]
train = train[:,1:]

pickle.dump(train, open("train.p","wb"))
pickle.dump(test, open("test.p","wb"))
pickle.dump(target, open("target.p","wb"))