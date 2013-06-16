import numpy as np
from datetime import datetime as dt
from sklearn import svm, metrics
from sklearn.cross_validation import KFold
from scipy import stats

with open('KGDR3.log','a') as f:
	X = np.genfromtxt('reducedtrain.csv', delimiter=',')
	Y = X[:,0]
	X = X[:,1:]/255
	f.write("\n\nStarting at %s\n" % dt.now())
	f.write("Data sets loaded and split\n")

	for C_val in [100, 300, 1000]:
		for gamma_val in [0.015, 0.0175, 0.02, 0.0225, 0.025]:

			kf = KFold(len(Y), n_folds=5, indices=False)
			classifier = [0]*5
			accuracy = [0]*5
			for model_num, (train, test) in enumerate(kf):

				X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]

				classifier[model_num] = svm.SVC(C=C_val,cache_size=750,gamma=gamma_val)

				classifier[model_num].fit(X_train, Y_train)

				expected = Y_test
				predicted = classifier[model_num].predict(X_test)
				accuracy[model_num] = metrics.accuracy_score(expected, predicted)

				f.write ("Accuracy with C = %f and gamma = %f on fold %i: %f\n" % (
						C_val, gamma_val, model_num+1, accuracy[model_num]))
			
			f.write ("%s\n" % classifier[model_num])
			f.write ("Overall accuracy with C = %f and gamma = %f: %f\n" % (
								C_val, gamma_val, np.mean(accuracy)))
			f.write("Finishing at %s\n" % dt.now())
