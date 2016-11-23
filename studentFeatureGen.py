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

import re

REGEX = re.compile(r",\s*")
def tokenize(text):
    return [tok.strip().lower() for tok in REGEX.split(text)]

train = pd.read_csv('/home/shady/intern/Student.csv')
train_cols = train.columns
tr = np.array(train)

pd.isnull(train['Profile']) = ""

for i in count(train.rows):
	if(pd.isnull(train[i,'Profile')):
		train[i,'Profile'] = " "

studentProfile = CountVectorizer(tokenizer=tokenize,min_df=1)
studentProfileT = studentProfile.fit_transform(train['Profile'])
studentProfileF = np.array(studentProfile.get_feature_names())
skillReqCol = intern_cols[13:]
#skillUnique = np.unique(np.concatenate((skillReqFeature,skillReqCol),axis=0))

internTypeV = CountVectorizer(min_df=1)

internCat = CountVectorizer(min_df=1)

student_data = pd.read_csv('Student.csv')
student_cols = student_data.columns
sData = np.array(student_data)

