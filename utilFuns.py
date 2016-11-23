preds = pd.read_csv('xgbSol1.csv',header=None)
preds = np.array(preds)[:,0]
test=pd.read_csv('test.csv')

ss=pd.read_csv('submission.csv')
preds = pd.DataFrame({test.columns[0]:test[test.columns[0]],test.columns[1]:test[test.columns[1]],ss.columns[2]:preds})

preds.to_csv('xgbSolSubmit1.csv',index=False)
preds = pd.DataFrame(columns=[test.columns[0:2],ss.columns[2]])

preds[test.columns[0:2]] = test[test.columns[0:2]]
preds[ss.columns[2]]=sol


print "writing to xgbSolWithUpcLen.csv"

preds['VisitNumber'] = ss['VisitNumber']
preds.set_index('VisitNumber', inplace=True)
preds.to_csv('xgbSolWithUpcFract65700.csv')
 
 
 

nn=[]
with open('nn51260Common.pkl', 'rb') as infile:
	nn=pickle.load(infile)


preds = predictTest(nn,test)
preds = preds[:,-1]
test=pd.read_csv('test.csv')
ss=pd.read_csv('submission.csv')
preds = pd.DataFrame({test.columns[0]:test[test.columns[0]],test.columns[1]:test[test.columns[1]],ss.columns[2]:preds})

preds.to_csv('nnSolSubmit3.csv',index=False)









import json

with open("kerasModelJson.txt") as json_file:
    json_data = json.load(json_file)
    
    

import yaml

stream = open("kerasModel.yml", "r")
docs = yaml.load_all(stream)



import pandas as pd
import numpy as np
import xgboost as xgb
import copy
import array
from sknn.mlp import Classifier, Layer
import cPickle as pickle
import numpy as np
import scipy.sparse
from scipy.sparse import hstack
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



o1=pd.read_csv('enSol11_lucifier_xgb1.csv')###0.6874
o2=pd.read_csv('xgbBaggingResults_155.csv')##0.6857
o3=pd.read_csv('p1_9_lucifier_xgb1.csv')###0.6827
o4=pd.read_csv('xgbBaggingResults15.csv')####0.68141



o1=np.array(o1)[:,-2]
o2=np.array(o2)[:,-2]
o3=np.array(o3)[:,-2]
o4=np.array(o4)[:,-2]


o=o1*0.4+o2*0.3+o3*0.2+o4*0.1

preds=o

test=pd.read_csv('../test.csv')

ss=pd.read_csv('../submission.csv')
preds = pd.DataFrame({test.columns[0]:test[test.columns[0]],test.columns[1]:test[test.columns[1]],ss.columns[2]:preds})

preds.to_csv('ensemble2.csv',index=False)

