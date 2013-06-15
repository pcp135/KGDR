import numpy as np
from sklearn import svm, metrics
from sklearn.cross_validation import KFold
from scipy import stats
import pylab as pl

X = np.genfromtxt('reducedtrain.csv', delimiter=',')
test_set = np.genfromtxt('test.csv', delimiter=',')
Y = X[:,0]
X = X[:,1:]/255
test_set /= 255
print "Data sets loaded and split"

kf = KFold(len(Y), n_folds=5, indices=False)
classifier = [0]*5
for model_num, (train, test) in enumerate(kf):

	X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]

	# Create a classifier: a support vector classifier
	classifier[model_num] = svm.SVC(C=100,cache_size=750,gamma=0.01)

	# We learn the digits on the first half of the digits
	classifier[model_num].fit(X_train, Y_train)

	# Now predict the value of the digit on the second half:
	expected = Y_test
	predicted = classifier[model_num].predict(X_test)

	print "Classification report for classifier on cross validation set %s:\n%s\n" % (
	    classifier[model_num], metrics.classification_report(expected, predicted))
	#print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)
	confused = predicted != expected
	modifier = 0
	for index, (image, label) in enumerate(zip(X_test[confused], predicted[confused]))):
		pl.subplot(2, 4, index +modifier+ 1)
		pl.axis('off')
		pl.imshow(image.reshape(28,28), cmap=pl.cm.gray_r, interpolation='nearest')
		pl.title('Prediction: %i' % label)
		if index == 8:
			pl.show()
			modifier-=8

pred=np.zeros(test_set.shape[0])
#val=np.zeros(X.shape[0])
for i, model in enumerate(classifier):
	#val=np.vstack((val, model.predict(X)))
	pred=np.vstack((pred, model.predict(test_set)))

#voted = stats.mode(val[1:6],axis=0)[0][0]
#print metrics.classification_report(Y, voted)
voted = stats.mode(pred[1:6],axis=0)[0][0]

f=open('result5.csv','w')
for value in voted:
	f.write('%s\n' % (int(value),))
f.close()