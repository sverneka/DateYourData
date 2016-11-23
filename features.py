import pandas as pd
import numpy as np
import xgboost as xgb
import copy
import array
from sknn.mlp import Classifier, Layer
import cPickle as pickle
import numpy as np
import scipy.sparse
from scipy.sparse import hstack, vstack
import cPickle as pickle
import numpy as np
import scipy.sparse
import pandas as pd
import array
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
from scipy.sparse import hstack
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

import re
REGEX = re.compile(r",\s*")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]


##############read all files#############
intern = pd.read_csv('Internship.csv')
intern = np.array(intern)
student = pd.read_csv('Student.csv')
student = np.array(student)
train= pd.read_csv('train.csv')
train = np.array(train)
test = pd.read_csv('test.csv')
test=np.array(test)



#############do some preprocessing to remove null and NaN and Not expected values##


#####################For Train files################################
#####prefered location of Train file##############
ind=[]
for i in xrange(0,train.shape[0]):
	if( pd.isnull(train[i,-3])):
		ind.append(i)


train[ind,-3]=0

#########expected stipend#################
ind=[]
for i in xrange(0,train.shape[0]):
	if(train[i,-5] == "No Expectations"):
		ind.append(i)


train[ind,-5] = 0

############Expected stipend take 1st digit###############
ind=[]
for i in xrange(0,train.shape[0]):
	a = str(train[i,-5])[0]
	train[i,-5]=a


train[:,-5] = train[:,-5].astype('float')

#################################For test files###################################
#####prefered location of test file##############
ind=[]
for i in xrange(0,test.shape[0]):
	if( pd.isnull(test[i,-2])):
		ind.append(i)


test[ind,-2]=0

#########expected stipend#################
ind=[]
for i in xrange(0,test.shape[0]):
	if(test[i,-4] == "No Expectations"):
		ind.append(i)


test[ind,-4] = 0

############Expected stipend take 1st digit###############
ind=[]
for i in xrange(0,test.shape[0]):
	a = str(test[i,-4])[0]
	test[i,-4]=a


test[:,-4] = test[:,-4].astype('float')





##################replace all null values with 0#########################
##########for intern##################
a=intern
for i in xrange(0,a.shape[0]):
	for j in xrange(0,a.shape[1]):
		if(pd.isnull(a[i,j])):
			a[i,j] = 0

#####for student#############
a=student
for i in xrange(0,a.shape[0]):
	for j in xrange(0,a.shape[1]):
		if(pd.isnull(a[i,j])):
			a[i,j] = 0


##############for train##############
a=train
for i in xrange(0,a.shape[0]):
	for j in xrange(0,a.shape[1]):
		if(pd.isnull(a[i,j])):
			a[i,j] = 0

############for test################
a=test
for i in xrange(0,a.shape[0]):
	for j in xrange(0,a.shape[1]):
		if(pd.isnull(a[i,j])):
			a[i,j] = 0



#####################location vectorizer##############################
loc_vec = CountVectorizer(min_df=1)
ind=np.where(intern[:,4].astype('str') == '0')[0]
intern[ind,4] = 'None'

ind=np.where(student[:,2].astype('str') == '0')[0]
student[ind,2] = 'None'

ind=np.where(student[:,-3].astype('str') == '0')[0]
student[ind,-3] = 'None'

ind=np.where(train[:,-3].astype('str') == '0')[0]
train[ind,-3] = 'None'

locMat = loc_vec.fit_transform(intern[:,4].astype('str').tolist()+student[:,2].astype('str').tolist()+student[:,3].astype('str').tolist()+student[:,-3].astype('str').tolist()+train[:,-3].astype('str').tolist())

internLocMat = scipy.sparse.csr_matrix(loc_vec.fit_transform(intern[:,4].astype('str').tolist()).astype('float'))
studentInstMat = scipy.sparse.csr_matrix(loc_vec.fit_transform(student[:,2].astype('str').tolist()).astype('float'))
studentHomMat = scipy.sparse.csr_matrix(loc_vec.fit_transform(student[:,3].astype('str').tolist()).astype('float'))
studentLoc = scipy.sparse.csr_matrix(loc_vec.fit_transform(student[:,-3].astype('str').tolist()).astype('float'))
temp = scipy.sparse.csr_matrix(loc_vec.fit_transform(train[:,-3].astype('str').tolist()+test[:,-2].astype('str').tolist()).astype('float'))

trainPrefLocMat = temp[0:train.shape[0],:]
testPreLocMat = temp[train.shape[0]:,:]


###############internship Profile###############
ind=np.where(intern[:,1].astype('str') == '0')[0]
intern[ind,1] = 'None'

ind=np.where(student[:,-4].astype('str') == '0')[0]
student[ind,-4] = 'None'


internProV = CountVectorizer(min_df=1,stop_words=set(('And','and','AND','or','Or','in','In')))
internPro=internProV.fit_transform(intern[:,1].astype('str').tolist()+student[:,-4].astype('str').tolist())

internPro = internProV.fit_transform(intern[:,1].astype('str').tolist())
studentPro = internProV.fit_transform(student[:,-4].astype('str').tolist())

skillV=CountVectorizer(tokenizer=tokenize,min_df=1)
skillReq = skillV.fit_transform(intern[:,2])
#skillUnique = np.unique(np.concatenate((skillReqFeature,skillReqCol),axis=0))
skillReq = scipy.sparse.csr_matrix(skillReq)

internTypeV = CountVectorizer(min_df=1)
internType=internTypeV.fit_transform(intern[:,3])
internType = scipy.sparse.csr_matrix(internType)

internCatV = CountVectorizer(min_df=1)
internCat=internCatV.fit_transform(intern[:,5])
internCat = scipy.sparse.csr_matrix(internCat)


internStiV = CountVectorizer(min_df=1)
internSti=internStiV.fit_transform(intern[:,7])
internSti = scipy.sparse.csr_matrix(internSti)

noOp = np.zeros((intern.shape[0],1))
noOp[:,0] = intern[:,6]
noOp = scipy.sparse.csr_matrix(noOp.astype('float')/100)

internId = np.zeros((intern.shape[0],1))
internId[:,0] = intern[:,0]
internId = scipy.sparse.csr_matrix(internId.astype('float'))

for i in xrange(0,student.shape[0]):
	if(student[i,1] == 'Y'):
		student[i,1] = 1
	else:
		student[i,1] = 0


ind=np.where(student[:,4].astype('str') == '0')[0]
student[ind,4] = 'None'
streamV = CountVectorizer(min_df=1,stop_words=set(('And','and','AND','or','Or','in','In')))
stream = scipy.sparse.csr_matrix(streamV.fit_transform(student[:,4]).astype('float'))

ind=np.where(student[:,5].astype('str') == '0')[0]
student[ind,5] = 'None'
degreeV = CountVectorizer(min_df=1,stop_words=set(('And','and','AND','or','Or','in','In')))
degree = scipy.sparse.csr_matrix(degreeV.fit_transform(student[:,5]).astype('float'))

ind=np.where(student[:,6].astype('str') == '0')[0]
student[ind,6] = 'None'
yearV = CountVectorizer(min_df=1,stop_words=set(('And','and','AND','or','Or','in','In')))
year = scipy.sparse.csr_matrix(degreeV.fit_transform(student[:,6]).astype('float'))


student[:,8] = student[:,8].astype('float')/student[:,9]
student[:,10] = student[:,10].astype('float')/student[:,11]
stuUG = np.zeros((student.shape[0],1)).astype('float')
stuUG[:,0] = student[:,10]

ind=np.where(student[:,-5].astype('str') == '0')[0]
student[ind,-5] = 'None'
expV = CountVectorizer(min_df=1,stop_words=set(('And','and','AND','or','Or','in','In')))
exp = scipy.sparse.csr_matrix(expV.fit_transform(student[:,-5]).astype('float'))



current=student[0,0]
currentId = 0
for i in xrange(0,student.shape[0]-1):
	next = student[i+1,0]
	if(current == next):
		student[currentId,-3] = str(student[currentId,-3])+" "+str(student[i+1,-3])
		student[currentId,-4] = str(student[currentId,-4])+" "+str(student[i+1,-4])
		student[currentId,-5] = str(student[currentId,-5])+" "+str(student[i+1,-5])
	else:
		current=student[i+1,0]
		currentId=i+1

student[:,7] = (student[:,7].astype('float')-2000)/20.0
studentMatTr = hstack((scipy.sparse.csr_matrix(student[:,0:2].astype('float')),studentInstMat,studentHomMat,degree,stream,year,scipy.sparse.csr_matrix(student[:,7:9].astype('float')),scipy.sparse.csr_matrix(stuUG),scipy.sparse.csr_matrix(student[:,11:13].astype('float')/100),exp,studentPro,studentLoc,scipy.sparse.csr_matrix(stuUG)))
studentMatTr = studentMatTr.toarray()

del(studentInstMat,studentHomMat,degree,stream,year,exp,studentPro,studentLoc)

urveshExp = pd.read_csv('studentExp.csv')
urveshExp = np.array(urveshExp)
urveshExp = urveshExp[:,1:]


studentDict={}
current=student[0,0]
currentId = 0
studentDict[current] = studentMatTr[currentId,1:]
for i in xrange(0,studentMatTr.shape[0]-1):
	next = student[i+1,0]
	if(current == next):
		continue
	else:
		current=student[i+1,0]
		currentId=i+1
		studentDict[current] = studentMatTr[currentId,1:]


for i in xrange(0,urveshExp.shape[0]):
	studentDict[urveshExp[i,0]][-1] = urveshExp[i,1]


################################create internship dictionary ############################################
intern[:,12] = intern[:,12].astype('float')/12
internMatTr = hstack((internId,internPro,skillReq,internType,internLocMat,internCat,noOp,internSti,scipy.sparse.csr_matrix(intern[:,8:10].astype('float')/50000),scipy.sparse.csr_matrix(intern[:,12:].astype('float'))))
internMatTr=internMatTr.toarray()
internDict={}
for i in xrange(0,internMatTr.shape[0]):
	internDict[internMatTr[i,0]] = internMatTr[i,1:]


del(internId,internPro,skillReq,internType,internLocMat,internCat,noOp,internSti)
#############################Create train file ######################################


trainExpSti = pd.read_csv('trainExpStipend.csv',index_col=False)
trainExpSti = scipy.sparse.csr_matrix(np.array(trainExpSti)[:,3:].astype('float'))
testExpSti = pd.read_csv('testExpStipend.csv',index_col=False)
testExpSti = scipy.sparse.csr_matrix(np.array(testExpSti)[:,3:].astype('float'))

isPartTimeTrain = np.zeros((train.shape[0],1))
isPartTimeTrain[:,0] = train[:,-2]
train[:,3] = train[:,3].astype('float')/10
train[:,4] = train[:,4].astype('float')/12

trainMatTr= hstack((train[:,3:5].astype('float'),trainPrefLocMat,isPartTimeTrain.astype('float'),trainExpSti))
del(trainPrefLocMat,isPartTimeTrain)
trainMatTr = trainMatTr.toarray()
i=0
a=scipy.sparse.csr_matrix(np.concatenate((internDict[train[i,0]],studentDict[train[i,1]],trainMatTr[i,:])))
trainMat=scipy.sparse.csr_matrix(np.zeros((1,a.shape[1])))
tempMat=scipy.sparse.csr_matrix(np.zeros((1,a.shape[1])))
label=np.zeros((train.shape[0],1))
for i in xrange(0,train.shape[0]):
	a=scipy.sparse.csr_matrix(np.concatenate((internDict[train[i,0]],studentDict[train[i,1]],trainMatTr[i,:])))
	tempMat=vstack((tempMat,a))
	label[i,0]=train[i,-1]
	if((i%10000==0) or (i==train.shape[0]-1)):
		trainMat = vstack((trainMat,tempMat[1:,:]))
		tempMat=scipy.sparse.csr_matrix(np.zeros((1,a.shape[1])))
		print i
	


trainMat = trainMat[1:,:]


with open('train_nn_sparse_mat.dat', 'wb') as outfile:
    pickle.dump(trainMat, outfile, pickle.HIGHEST_PROTOCOL)



labels = scipy.sparse.csr_matrix(label)
with open('label_sparse_mat.dat', 'wb') as outfile:
    pickle.dump(labels, outfile, pickle.HIGHEST_PROTOCOL)

#############################create test file#####################################
isPartTimeTest = np.zeros((test.shape[0],1))
isPartTimeTest[:,0] = test[:,-1]
test[:,3] = test[:,3].astype('float')/10
test[:,4] = test[:,4].astype('float')/12
testMatTe= hstack((test[:,3:5].astype('float'),testPreLocMat,isPartTimeTest.astype('float'),testExpSti))
#del(testPrefLocMat,isPartTimeTest)
testMatTe = testMatTe.toarray()
i=0
a=scipy.sparse.csr_matrix(np.concatenate((internDict[test[i,0]],studentDict[test[i,1]],testMatTe[i,:])))
testMat=scipy.sparse.csr_matrix(np.zeros((1,a.shape[1])))
tempMat=scipy.sparse.csr_matrix(np.zeros((1,a.shape[1])))
for i in xrange(0,test.shape[0]):
	a=scipy.sparse.csr_matrix(np.concatenate((internDict[test[i,0]],studentDict[test[i,1]],testMatTe[i,:])))
	tempMat=vstack((tempMat,a))
	label[i,0]=test[i,-1]
	if((i%10000==0) or (i==test.shape[0]-1)):
		testMat = vstack((testMat,tempMat[1:,:]))
		tempMat=scipy.sparse.csr_matrix(np.zeros((1,a.shape[1])))
		print i
	


testMat = testMat[1:,:]


testMat = scipy.sparse.csr_matrix(testMat.astype('float'))
with open('test_sparse_mat.dat', 'wb') as outfile:
    pickle.dump(testMat, outfile, pickle.HIGHEST_PROTOCOL)




































