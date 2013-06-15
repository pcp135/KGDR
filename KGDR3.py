import numpy as np
from sklearn import svm, metrics
from sklearn.cross_validation import KFold
from scipy import stats

X = np.genfromtxt('reducedtrain.csv', delimiter=',')
Y = X[:,0]
X = X[:,1:]/255
print "Data sets loaded and split"

kf = KFold(len(Y), n_folds=5, indices=False)
classifier = [0]*5
for model_num, (train, test) in enumerate(kf):

	X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]

	classifier[model_num] = svm.SVC(C=100,cache_size=750,gamma=0.01)

	# We learn the digits on the first half of the digits
	classifier[model_num].fit(X_train, Y_train)

	# Now predict the value of the digit on the second half:
	expected = Y_test
	predicted = classifier[model_num].predict(X_test)

	print "Classification report for classifier on cross validation set %s:\n%s\n" % (
	    classifier[model_num], metrics.classification_report(expected, predicted))
	print "Overall accuracy with C = %f and gamma = %f: %f" % (
			C_val, gamma_val, metrics.accuracy_score(expected, predicted))
	#print "Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)

