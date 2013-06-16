import numpy as np
from datetime import datetime as dt
from sklearn import svm, metrics
from sklearn.cross_validation import KFold
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

with open('KGDR4.log','a') as f:
	X = np.genfromtxt('reducedtrain.csv', delimiter=',')
	Y = X[:,0]
	X = X[:,1:]/255
	f.write("\n\nStarting at %s\n" % dt.now())
	f.write("Data sets loaded and split\n")
	
	for ests in [10,30,60,100,300]:
		for feats in [10,20,28,30,40]:
			kf = KFold(len(Y), n_folds=5, indices=False)
			classifier = [0]*5
			accuracy = [0]*5
			for model_num, (train, test) in enumerate(kf):

				X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]

				classifier[model_num] = RandomForestClassifier(
					n_estimators=ests,max_features=feats)

				classifier[model_num].fit(X_train, Y_train)

				expected = Y_test
				predicted = classifier[model_num].predict(X_test)
				accuracy[model_num] = metrics.accuracy_score(expected, predicted)

				f.write ("Accuracy on fold %i with estimators = %i, features = %i: %f\n" % (
					model_num+1, ests, feats, accuracy[model_num]))
	
			f.write ("%s\n" % classifier[model_num])
			f.write ("Overall accuracy with estimators = %i, features = %i: %f\n" % (
				ests, feats, np.mean(accuracy)))
	f.write("Finishing at %s\n" % dt.now())
