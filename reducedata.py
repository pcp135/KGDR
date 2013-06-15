f = open('train.csv',"r")
lines = [line.strip() for line in f]
for line in lines[:5000]:
	print line