import numpy as np
import pandas as pd
from scipy.special import expit
import random
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding, MDS
from sklearn.decomposition import TruncatedSVD
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.cross_validation import train_test_split
import cPickle as pickle
import scipy.sparse
import array
from sknn.mlp import Classifier, Layer
from scipy.sparse import hstack
import xgboost as xgb

path='/home/shady/intern'

random.seed(21)
np.random.seed(21)

def load_train_data(path,modelNo,depth,eta,rounds,ratio=0):
    X=[]
    with open(path+'/train_sparse_mat.dat', 'rb') as infile:
        X = pickle.load(infile)
    random.seed(modelNo)
    np.random.seed(modelNo)
    labels=[]
    with open(path+'/label_sparse_mat.dat', 'rb') as infile:
    	labels = pickle.load(infile)
    labels=labels.toarray().astype('int')
    y=labels
    #X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=modelNo,stratify=y)
    w=labels.astype('float')
    w[:]=1.0
    if(ratio>0):
	print "sachin"
    	ind=np.where(labels==1)[0]
    	w[ind]=ratio
    	ind=np.where(labels==0)[0]
    	w[ind]=1-ratio
    print "weights are ",w[0:10]
    xgtrain = xgb.DMatrix(X,y,weight=w)
    #xgval = xgb.DMatrix(X_val,y_val)
    watchlist = [(xgtrain, 'train')]
    params = {"objective": "binary:logistic",
              "eta": 0.01,# used to be 0.2 or 0.1
              "max_depth": 6, # used to be 5 or 6
              "min_child_weight": 1,
              "max_delta_step": 6,
              "silent": 1,
              "colsample_bytree": 0.7,
                      "subsample": 0.8,
                      "eval_metric" : "auc",
              "seed": 1}
    params['max_depth']=depth
    params['eta']=eta
    plst = list(params.items())
    num_rounds = rounds
    rbm1 = xgb.train(params, xgtrain, num_rounds, watchlist)
    #rbm1 = SVC(C=100.0, gamma = 0.1, probability=True, verbose=1).fit(X[0:9999,:], y[0:9999])
    #rbm2 = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features='auto', bootstrap=False, oob_score=False, n_jobs=1, verbose=1).fit(X[0:9999,:], y[0:9999])
    #rbm3 = GradientBoostingClassifier(n_estimators=50,max_depth=11,subsample=0.8,min_samples_leaf=5,verbose=1).fit(X[0:9999,:], y[0:9999])
    with open(path+'/test_sparse_mat.dat', 'rb') as infile:
        Y = pickle.load(infile)
    xgtest=xgb.DMatrix(Y)
    preds1=rbm1.predict(xgtest)
    print preds1.shape
    test=pd.read_csv(path+'/test.csv')
    p1 = pd.DataFrame({test.columns[0]:test[test.columns[0]],test.columns[1]:test[test.columns[1]],ss.columns[2]:preds1})
    p1.to_csv("p1_"+str(modelNo)+".csv",index=False)
    temp_p[:,0] = preds1
    return temp_p

num_runs = 10
test=[]
with open(path+'/test_sparse_mat.dat', 'rb') as infile:
	test = pickle.load(infile)


y_prob = np.zeros((test.shape[0],1))
temp_p = np.zeros((test.shape[0],1))
ss=pd.read_csv(path+'/submission.csv')

grid=[[15,0.02,800,0.6],[10,0.02,2000,0.6],[10,0.02,1500,0.6],[10,0.02,2500,0.6],[8,0.01,5000,0.6],[5,0.005,10000,0.8],[5,0.01,5000,0.8],[5,0.005,13000,0.6],[6,0.005,12000,0.6],[6,0.01,8000,0.8]]
for jj in xrange(num_runs):
  print(jj)
  preds = load_train_data(path,jj+1,grid[jj][0],grid[jj][1],grid[jj][2],grid[jj][3])
  print y_prob.shape
  y_prob = y_prob + preds
  preds = y_prob/(jj+1.0)
  preds = preds[:,0]
  test=pd.read_csv(path+'/test.csv')
  preds = pd.DataFrame({test.columns[0]:test[test.columns[0]],test.columns[1]:test[test.columns[1]],ss.columns[2]:preds})
  preds.to_csv('enSol'+str(jj+1)+'.csv',index=False)


y_prob = y_prob/(num_runs-1+1.0)

preds = y_prob[:,0]
test=pd.read_csv(path+'/test.csv')
preds = pd.DataFrame({test.columns[0]:test[test.columns[0]],test.columns[1]:test[test.columns[1]],ss.columns[2]:preds})

print "writing to xgbSolWithUpcLen.csv"
preds.to_csv('xgbBaggingResults.csv',index=False)
