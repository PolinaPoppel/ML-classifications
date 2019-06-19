import numpy as np
import random
import math
import csv
from numpy import genfromtxt


#import matplotlib.pyplot as plt
#np.set_printoptions(threshold=np.nan)


def readTXT(filename):
	file = open(filename, "r") 
	make_list  = file.read().split(',') 
	make_list.pop()
	for i in range (0, len(make_list)):
		make_list[i] = float(make_list[i])
	return make_list

def covNN(filename):
	matrix = []
	file = open(filename, "r") 
	
	for i in range (0, 20):
		line = file.readline()
		line = line.split(',') 
		line.pop()
		for j in range (0, len(line)):
			line[j] = float(line[j])
		matrix.append(line)
	matrix = np.array(matrix)

	return matrix


m_0 = readTXT("DS1_m_0.txt")
m_1 = readTXT("DS1_m_1.txt")
covM = covNN("DS1_Cov.txt")

x1 = np.random.multivariate_normal(m_0, covM, 2000)
x1 = np.insert(x1, 0, values=0, axis=1)
x2 = np.random.multivariate_normal(m_1, covM, 2000)
x2 = np.insert(x2, 0, values=1, axis=1)
#print (x2)

np.random.shuffle(x1)
test1, temp1 = x1[:400,:], x1[400:,:]

np.random.shuffle(temp1)
valid1, training1 = temp1[:400,:], temp1[400:,:]

np.random.shuffle(x2)
test2, temp2 = x2[:400,:], x2[400:,:]

np.random.shuffle(temp2)
valid2, training2 = temp2[:400,:], temp2[400:,:]


train = np.append(training1, training2, axis=0)
test = np.append(test1, test2, axis=0)
valid = np.append(valid1, valid2, axis=0)

np.random.shuffle(train)
np.random.shuffle(test)

#target = test[:,0]
#x_test  = test[:,1:]

#np.savetxt("DS1train.csv", train, delimiter=",")
#np.savetxt("DS1valid.csv", valid, delimiter=",")
#np.savetxt("DS1test.csv", test, delimiter=",")

train = genfromtxt('DS1train.csv', delimiter=',')
valid = genfromtxt('DS1valid.csv', delimiter=',')
test = genfromtxt('DS1test.csv', delimiter=',')
#print (test.shape)
#print ("test shape: :", test.shape)
#print ("target shape: :", target.shape)
#print ("test  x shape: :", x_test.shape)
################################################################

#estimanting parameters for GDA
#isolate t_n
target = test[:,0]
x_test  = test[:,1:]
t_n = train[:,0]
x_n  = train[:,1:]

t_valid = valid[:,0]
x_valid  = valid[:,1:]

p_C0 = 0.5
p_C1 = 0.5
x_n  = train[:,1:]
N1 = int(train.shape[0]/2)
N2 = N1
#print("N1: ", N1)
#print("N2: ", N1)
#print(len(t_n))

summation = np.zeros([1, 21])
for i in  train:
	if(i[0] == 0):
		#print(summation)
		summation = np.add(summation, i)

mu1 = (1/N1)*summation

summation2 = np.zeros([1, 21])
for i in  train:
	if(i[0] == 1):
		summation2 = np.add(summation2, i)

mu2 = (1/N2)*summation2
mu1 = mu1[:,1:]
mu2 = mu2[:,1:]
mu1 = np.transpose(mu1)
mu2 = np.transpose(mu2)

#print("mu1: ", mu1)
#print("mu2: ", mu2)

#print(mu2)
#print(mu2.shape)
#print(x_n.shape)

#evaluating S1
summ1 = np.zeros([20, 20])
j = np.zeros([1, 20])
summ2 = np.zeros([20, 20])
jj = np.zeros([1, 20])
cnt = 0
for i in x_n:
	if (t_n[cnt]== 0):
		#print(i.shape)
		i = np.add(i, j)
		#print(i.shape)
		i = np.transpose(i)
		#print(i.shape)
		a = np.subtract(i, mu1)
		b = np.transpose(a)
		c = np.matmul(a, b)
		summ1 = np.add(summ1, c)

	else:
		i = np.add(i, jj)
		#print(i.shape)
		i = np.transpose(i)
		#print(i.shape)
		a = np.subtract(i, mu2)
		b = np.transpose(a)
		c = np.matmul(a, b)
		summ2 = np.add(summ2, c)
		#print(cnt)
	cnt = cnt + 1

S1 = (1/N1)*summ1

S2 = (1/N2)*summ2
#print("S2: ", S2)


temp1_sigma = 0.5*S1
temp2_sigma = 0.5*S2
sigma = np.add(temp1_sigma, temp2_sigma)
#print("sigma: ", sigma)
sigma_inv = np.linalg.inv(sigma)
w_temp = np.subtract(mu1, mu2)

w = np.matmul(sigma_inv, w_temp)

#print("w ", w)
p1 = np.matmul(np.transpose(mu1), sigma_inv)
p2 = (-0.5)*(np.matmul(p1, mu1))

p3 = np.matmul(np.transpose(mu2), sigma_inv)
p4 = (0.5)*(np.matmul(p3, mu2))

w0 = p2 + p4
#print("w0: ", w0)

def sigmoid(a):
	value =  1/(1+np.exp(-1*a))
	return value

probs = []	
for j in x_test:
	a = np.matmul(np.transpose(w), j) + w0
	prob = sigmoid(a)
	if(prob >= 0.5):
		probs.append(0)
	else:
		probs.append(1)
	#probs.append(prob)
#print("probs: ", len(probs))
#print(target.shape)

acc = 0
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
for i in range (0, len(probs)):
	if(probs[i] == target[i]):
		if(probs[i] == 1):
			true_pos = true_pos + 1
		else:
			true_neg = true_neg+1
		acc=acc+1
	else:
		if(probs[i] == 1):
			false_pos = false_pos +1
		else:
			false_neg = false_neg +1

#print("acc ", acc)
accuracy = acc/len(probs)
print("accuracy: ", accuracy)
print("true_pos ", true_pos)
print("true_neg ", true_neg)
print("false_pos ", false_pos)
print("false_neg ", false_neg)

precision = true_pos/(true_pos+false_pos)
recall = true_pos/(true_pos+false_neg)
print(precision)
print(recall)
F1 = (2*precision*recall)/(precision+recall)
print(F1)

#np.savetxt('Assignment2_260606583_2_1_mu1DS1.txt', mu1, delimiter=',')
#np.savetxt('Assignment2_260606583_2_1_mu2DS1.txt', mu2, delimiter=',')
#np.savetxt('Assignment2_260606583_2_1_sigmaDS1.txt', sigma, delimiter=',')
#np.savetxt('Assignment2_260606583_2_1_w.txt', w, delimiter=',')
#with open('Assignment2_260606583_2_1_w0.txt', 'w') as f:
#  f.write('%d' % w0)

#################


'''

#print(x_n[0])
def dist_matrix(array, target):
	dist_m = []
	for x in array:
		distances = []
		index = 0
		for point in array: 
			diff = np.subtract(x, point)
			summ = np.linalg.norm(diff)
			dist = math.sqrt(summ)
			tupl = [target[index], dist]
			distances.append(tupl)
			index+=1
		distances = sorted(distances, key=lambda a_entry: a_entry[1]) 
		dist_m.append(distances)
	return dist_m

#train_dist = dist_matrix(x_n, t_n) 
#valid_dist = dist_matrix(x_valid, t_valid) 
#test_dist = dist_matrix(x_test, target)
#print(test_dist)


#selecting k


def accuracy(dist_matrix, actual, k):
	matrix = []
	for point in dist_matrix:
		point = point[1:]
		selected = point[:k]
		matrix.append(selected)

	#print(matrix)
	result = []
	for point in matrix:
		count_0 = 0
		count_1 = 0
		for tupl in point:
			if(tupl[0] == 0):
				count_0+=1
			else: 
				count_1+=1
		if(count_0 >= count_1):
			result.append(0)
		else:
			result.append(1)

	acc = 0
	true_pos = 0
	true_neg = 0
	false_pos = 0
	false_neg = 0
	#print(result)
	for i in range(0, len(result)):
		if(actual[i] == result[i]):
			acc+=1
			if(result[i] == 1):
				true_pos = true_pos + 1
			else:
				true_neg = true_neg+1
		else:
			if(result[i] == 1):
				false_pos = false_pos +1
			else:
				false_neg = false_neg +1
	accuracy_k = acc/len(result)
	precision = true_pos/(true_pos+false_pos)
	recall = true_pos/(true_pos+false_neg)
	F1 = (2*precision*recall)/(precision+recall)
	return [accuracy_k, precision, recall, F1]

F1s_train = []
F1s_valid = []
F1s_test = []

#for i in range (1, 20):
#	k = 10*i
#	F1s_train.append(accuracy(train_dist, t_n, k)[3])
	#F1s_test.append(accuracy(test_dist, target, k)[3])
#print(F1s_train)

#for i in range (8, 12):
#	k = 10*i
#	F1s_valid.append(accuracy(valid_dist, t_valid, k)[3])
#print(F1s_valid)

k = 90
print("test: ")
#print(accuracy(test_dist, target, 90))



'''

###############################################

c1_m1 = readTXT("DS2_c1_m1.txt")
c1_m2 = readTXT("DS2_c1_m2.txt")
c1_m3 = readTXT("DS2_c1_m3.txt")
cov1 = covNN("DS2_Cov1.txt")

c2_m1 = readTXT("DS2_c2_m1.txt")
c2_m2 = readTXT("DS2_c2_m2.txt")
c2_m3 = readTXT("DS2_c2_m3.txt")
cov2 = covNN("DS2_Cov2.txt")
cov3 = covNN("DS2_Cov3.txt")

x1_1 = np.random.multivariate_normal(c1_m1, cov1, 2000)
x1_1 = np.insert(x1_1, 0, values=0, axis=1)
x2_1 = np.random.multivariate_normal(c1_m2, cov2, 2000)
x2_1 = np.insert(x2_1, 0, values=0, axis=1)
x3_1 = np.random.multivariate_normal(c1_m3, cov3, 2000)
x3_1 = np.insert(x3_1, 0, values=0, axis=1)
np.random.shuffle(x1_1)
np.random.shuffle(x2_1)
np.random.shuffle(x3_1)

class1_p1 = x1_1[:200,:]
class1_p2 = x2_1[:840,:]
class1_p3 = x3_1[:960,:]

class1_temp = np.append(class1_p1, class1_p2, axis=0)
class1 = np.append(class1_temp, class1_p3, axis=0)
np.random.shuffle(class1)
#print(class1.shape)


x1_2 = np.random.multivariate_normal(c1_m1, cov1, 2000)
x1_2 = np.insert(x1_2, 0, values=1, axis=1)
x2_2 = np.random.multivariate_normal(c1_m2, cov2, 2000)
x2_2 = np.insert(x2_2, 0, values=1, axis=1)
x3_2 = np.random.multivariate_normal(c1_m3, cov3, 2000)
x3_2 = np.insert(x3_2, 0, values=1, axis=1)
np.random.shuffle(x1_2)
np.random.shuffle(x2_2)
np.random.shuffle(x3_2)

class2_p1 = x1_2[:200,:]
class2_p2 = x2_2[:840,:]
class2_p3 = x3_2[:960,:]

class2_temp = np.append(class2_p1, class2_p2, axis=0)
class2 = np.append(class2_temp, class2_p3, axis=0)
np.random.shuffle(class2)
#print(class2)


test_c1, temp_c1 = class1[:400,:], class1[400:,:]
np.random.shuffle(temp_c1)
valid_c1, train_c1 = temp_c1[:400,:], temp_c1[400:,:]
#print(test_c1.shape)
#print(valid_c1.shape)
#print(train_c1.shape)
test_c2, temp_c2 = class2[:400,:], class2[400:,:]
np.random.shuffle(temp_c2)
valid_c2, train_c2 = temp_c2[:400,:], temp_c2[400:,:]

train_DS2 = np.append(train_c1, train_c2, axis=0)
test_DS2 = np.append(test_c1, test_c2, axis=0)
valid_DS2 = np.append(valid_c1, valid_c2, axis=0)

np.random.shuffle(train_DS2)
np.random.shuffle(test_DS2)
np.random.shuffle(valid_DS2)
#print(train_DS2)
#print(test_DS2)
#print(valid_DS2)

#np.savetxt("DS2_train.csv", train_DS2, delimiter=",")
#np.savetxt("DS2_valid.csv", valid_DS2, delimiter=",")
#np.savetxt("DS2_test.csv", test_DS2, delimiter=",")

train_DS2 = genfromtxt('DS2_train.csv', delimiter=',')
valid_DS2 = genfromtxt('DS2_valid.csv', delimiter=',')
test_DS2 = genfromtxt('DS2_test.csv', delimiter=',')

#print(my_data.shape)

######################################################


t_DS2 = train_DS2[:,0]
x_train2  = train_DS2[:,1:]

valid_t_DS2 = valid_DS2[:,0]
x_valid2  = valid_DS2[:,1:]

target_DS2 = test_DS2[:,0]
x_test_DS2  = test_DS2[:,1:]



N1 = int(train_DS2.shape[0]/2)
N2 = N1

summ_DS2 = np.zeros([1, 21])
for i in  train_DS2:
	if(i[0] == 0):
		summ_DS2 = np.add(summ_DS2, i)
mu1_DS2 = (1/N1)*summ_DS2

summ2_DS2 = np.zeros([1, 21])
for i in  train_DS2:
	if(i[0] == 1):
		summ2_DS2 = np.add(summ2_DS2, i)
mu2_DS2 = (1/N2)*summ2_DS2


mu1_DS2 = mu1_DS2[:,1:]
mu2_DS2 = mu2_DS2[:,1:]
mu1_DS2 = np.transpose(mu1_DS2)
mu2_DS2 = np.transpose(mu2_DS2)

#print(mu1_DS2.shape)
#print(mu1_DS2)
#print(mu2_DS2.shape)
#print(mu2_DS2)
#print(x_n.shape)


summ_DS2 = np.zeros([20, 20])
j = np.zeros([1, 20])
summ2_DS2 = np.zeros([20, 20])
jj = np.zeros([1, 20])
cnt = 0
for i in x_train2:
	if (t_DS2[cnt]== 0):
		i = np.add(i, j)
		i = np.transpose(i)
		a = np.subtract(i, mu1_DS2)
		b = np.transpose(a)
		c = np.matmul(a, b)
		summ_DS2 = np.add(summ_DS2, c)

	else:
		i = np.add(i, jj)
		i = np.transpose(i)
		a = np.subtract(i, mu2_DS2)
		b = np.transpose(a)
		c = np.matmul(a, b)
		summ2_DS2 = np.add(summ2_DS2, c)
	cnt = cnt + 1

S1_DS2 = (1/N1)*summ_DS2

S2_DS2 = (1/N2)*summ2_DS2

print(S1_DS2.shape)

temp1_sigma_DS2 = 0.5*S1_DS2
temp2_sigma_DS2 = 0.5*S2_DS2
sigma_DS2 = np.add(temp1_sigma_DS2, temp2_sigma_DS2)

sigma_inv_DS2 = np.linalg.inv(sigma_DS2)
w_temp_DS2 = np.subtract(mu1_DS2, mu2_DS2)

w_DS2 = np.matmul(sigma_inv_DS2, w_temp_DS2)


p1_DS2 = np.matmul(np.transpose(mu1_DS2), sigma_inv_DS2)
p2_DS2 = (-0.5)*(np.matmul(p1_DS2, mu1_DS2))

p3_DS2 = np.matmul(np.transpose(mu2_DS2), sigma_inv_DS2)
p4_DS2 = (0.5)*(np.matmul(p3_DS2, mu2_DS2))

w0_DS2 = p2_DS2 + p4_DS2

#print(w_DS2, w_DS2.shape)
#print(w0_DS2)

probs_DS2 = []	
for j in x_test_DS2:
	a = np.matmul(np.transpose(w_DS2), j) + w0_DS2
	prob = sigmoid(a)
	if(prob >= 0.5):
		probs_DS2.append(0)
	else:
		probs_DS2.append(1)
	#probs.append(prob)

print(len(probs_DS2))

acc_DS2 = 0
true_pos_DS2 = 0
true_neg_DS2 = 0
false_pos_DS2 = 0
false_neg_DS2 = 0
for i in range (0, len(probs_DS2)):
	if(probs_DS2[i] == target_DS2[i]):
		#print(probs[i])
		if(probs_DS2[i] == 1):
			true_pos_DS2 = true_pos_DS2 + 1
		else:
			true_neg_DS2 = true_neg_DS2 + 1
		acc_DS2=acc_DS2+1
	else:
		if(probs_DS2[i] == 1):
			false_pos_DS2 = false_pos_DS2 +1
		else:
			false_neg_DS2 = false_neg_DS2 +1

#print("acc ", acc)
accuracy_DS2 = acc_DS2/len(probs_DS2)
print("accuracy_DS2: ", accuracy_DS2)
print("true_pos_DS2 ", true_pos_DS2)
print("true_neg_DS2 ", true_neg_DS2)
print("false_pos_DS2 ", false_pos_DS2)
print("false_neg_DS2 ", false_neg_DS2)

precision_DS2 = true_pos_DS2/(true_pos_DS2+false_pos_DS2)
recall_DS2 = true_pos_DS2/(true_pos_DS2+false_neg_DS2)
print(precision_DS2)
print(recall_DS2)
F1_DS2 = (2*precision_DS2*recall_DS2)/(precision_DS2+recall_DS2)
print(F1_DS2)

np.savetxt('Assignment2_260606583_5_1_mu1DS2.txt', mu1_DS2, delimiter=',')
np.savetxt('Assignment2_260606583_5_1_mu2DS2.txt', mu2_DS2, delimiter=',')
np.savetxt('Assignment2_260606583_5_1_sigmaDS2.txt', sigma_DS2, delimiter=',')
np.savetxt('Assignment2_260606583_5_1_w.txt', w_DS2, delimiter=',')
with open('Assignment2_260606583_5_1_w0.txt', 'w') as f:
  f.write('%d' % w0_DS2)

'''
###############################################
F1s_train_DS2 = []
F1s_valid_DS2 = []
F1s_test_DS2 = []

train_dist_DS2 = dist_matrix(x_train2, t_DS2) 
valid_dist_DS2 = dist_matrix(x_valid2, valid_t_DS2) 
test_dist_DS2 = dist_matrix(x_test_DS2, target)

for i in range (1, 20):
	k = 10*i
	F1s_train_DS2.append(accuracy(train_dist_DS2, t_DS2, k)[3])
	#F1s_valid_DS2.append(accuracy(test_dist, target, k)[3])
print(F1s_train_DS2)

for i in range (8, 15):
	k = 10*i
	F1s_valid_DS2.append(accuracy(valid_dist_DS2, valid_t_DS2, k)[3])
print(F1s_valid_DS2)

k = 80
print("test_DS2: ")
print(accuracy(test_dist_DS2, target_DS2, k))
#for point in train_dist:
	#iterating thru all distances (tuples!!) for a certain pt
#	point = sorted(point, key=lambda a_entry: a_entry[1]) 
#print(train_dist)

'''
