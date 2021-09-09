import numpy as np
import pandas as pd
from collections import defaultdict
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
import itertools


normal = pd.read_csv('data/normal.csv')
normal['label'] = 0
#print(normal.describe())

mining = pd.read_csv('data/mining.csv')
mining['label'] = 1
#print(mining.describe())

def generate_feature(data):
    data['TimeBin'] = data['_ws.col.Time']
    data['TimeBin'] = (data['TimeBin']/ 0.25 )
    data['TimeBin'] = data['TimeBin'].apply(np.floor)
    data['Direction'] = 0 # Client -> Server
    data['packet_ratio'] = 1000
    data['length_ratio'] = 1000
    data.loc[(
                (data['_ws.col.Source'] != '192.168.12.53') & 
                (data['_ws.col.Source'] != '192.168.12.62') &
                (data['_ws.col.Source'] != '192.168.12.108') &
                (data['_ws.col.Source'] != '192.168.12.243') &
                (data['_ws.col.Source'] != '192.168.12.244') &
                (data['_ws.col.Source'] != '172.20.10.2')
            ), ['Direction']] = 1 # Server -> Client
    group = data.groupby(['TimeBin'])
    group_features = group.apply(group_feature_extractor)
    group_features['delta_num_ttl'] = group_features['num_ttl'].diff(periods=1)
    group_features['delta_num_ttl'] = group_features['delta_num_ttl'].fillna(0)
    data = data.merge(group_features, left_on='TimeBin', right_index=True)

    group2 = data.groupby(['TimeBin','Direction'])
    group2_features = group2.apply(group2_feature_extractor)
    data = data.merge(group2_features, left_on=['TimeBin','Direction'], right_index=True)
    data.loc[(data['Direction'] == 1), 'c2s_packet_num'] = data['num_packet'] - data['c2s_packet_num']
    data.loc[(data['c2s_packet_num'] != 0), 'packet_ratio'] =  (data['num_packet'] - data['c2s_packet_num']) / data['c2s_packet_num']
    data.loc[(data['Direction'] == 1), 'c2s_length_sum'] = data['sum_length'] - data['c2s_length_sum']
    data.loc[(data['c2s_length_sum'] != 0), 'length_ratio'] =  (data['sum_length'] - data['c2s_length_sum']) / data['c2s_length_sum']
    return data
    
def group_feature_extractor(g):
    mean_length = (g['_ws.col.Length']).mean()
    std_length = (g['_ws.col.Length']).std()
    sum_length = (g['_ws.col.Length']).sum()
    num_packet = (g['_ws.col.No.']).count()
    num_ttl = len(set(g['_ws.col.TTL']))
    return pd.Series([mean_length, std_length, sum_length, num_packet,num_ttl], \
                        index = ['mean_length', 'std_length', 'sum_length', 'num_packet','num_ttl'])

def group2_feature_extractor(g):
    c2s_num_packet = (g['_ws.col.No.']).count()
    c2s_sum_length = (g['_ws.col.Length']).sum()
    return pd.Series([c2s_num_packet,c2s_sum_length], index = ['c2s_packet_num','c2s_length_sum'])


mining = generate_feature(mining)
#print(mining.loc[0:20,('_ws.col.No.','_ws.col.Time','_ws.col.Source','_ws.col.Destination','_ws.col.Length','Direction')])
normal = generate_feature(normal)
#print(mining.describe())


def generate_features_and_labels(data):
    #data.sort_values(['Time'],ascending=[1]) # sort all traffic by time
    data = data.dropna() # drop rows with either missing source or destination ports
    data = data.reset_index(drop=True)
    
    # GENERATE FEATURES
    features = data.copy(deep=True)

    # velocity, acceleration, and jerk in time in between successive packets
    features['dT'] = features['_ws.col.Delta_time']
    #features['dT2'] = features['dT'] - features['dT'].shift(1)
    #features['dT3'] = features['dT2'] - features['dT2'].shift(1)
    features = features.fillna(0) # fill offset rows with zeros #### FIX THIS - not working...

    # one hot encoding of common protocols: HTTP: TCP, UDP, and OTHER
    features['is_TCP'] = 0
    features.loc[features['_ws.col.Protocol'] == 'TCP', ['is_TCP']] = 1

    # GENERATE LABELS
    labels = features['label']

    del features['_ws.col.No.']
    del features['_ws.col.Time']
    del features['_ws.col.Source']
    del features['_ws.col.Destination']
    del features['_ws.col.Protocol']
    del features['_ws.col.Length']
    del features['_ws.col.Src_port']
    del features['_ws.col.Dst_port']
    del features['_ws.col.Delta_time'] #keep features['dT']
    del features['_ws.col.TTL']
    del features['label']
    del features['TimeBin']
    del features['Direction']
    del features['c2s_packet_num']
    del features['c2s_length_sum']
    del features['packet_ratio']
    del features['std_length']
    del features['num_packet']
    del features['sum_length']
    #-----------------------------
    #del features['length_ratio']
    #del features['num_ttl']
    #del features['delta_num_ttl']
    #del features['mean_length']
    #del features['dT']
    #del features['is_TCP']
    
    return (features, labels)

mining_features, mining_labels = generate_features_and_labels(mining)
normal_features, normal_labels = generate_features_and_labels(normal)
#print(mining_features.head(20))
#print(mining_labels.head(10))

all_features = pd.concat([mining_features, normal_features])
#print(all_features.columns)
all_labels = pd.concat([mining_labels, normal_labels])

#--------------Feature engineer-------------------
# all_features = MinMaxScaler().fit_transform(all_features)
# model = SelectKBest(chi2, k=7)
# model.fit_transform(all_features, all_labels)
# print(model.scores_)

# top6_features = list(all_features.columns[0:6])
# #print(top10_features)
# corr = all_features[top6_features].corr()
# plt.figure(figsize=(6,6))
# sns.heatmap(corr,annot=True,cmap='Greens')
# plt.show()


def sig(x):
    return 1/(1+np.exp(-x))

# classifies the model on training data and returns zero-one loss on test data
def classify(model, x_train, x_test, y_train, y_test):
    classifier = model
    if classifier.__class__.__name__ == "MultinomialNB":
        classifier.fit(sig(x_train),y_train)
    else:
        classifier.fit(x_train,y_train)
    y_predict = classifier.predict(x_test)
    
    # ANALYSIS: 
    print("==================================")
    print(classifier.__class__.__name__ + ":")
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_predict)  
    error = zero_one_loss(y_test, y_predict,)
    accuracy = 1 - error
    
    print("Normal Precision: " + str(precision[0]))
    print("Mining Precision: " + str(precision[1]))
    print("Normal Recall: " + str(recall[0])) 
    print("Mining Recall: " + str(recall[1])) 
    print("Normal F1: " + str(f1[0]))
    print("Mining F1: " + str(f1[1]))
    print("Error " + str(error))
    print("Accuracy " + str(accuracy))

    # confusion matrix    
    plt.figure()
    classes = ['Normal', 'Mining']
    cm = confusion_matrix(y_test, y_predict)
    np.set_printoptions(precision=2)  #设置输出精度

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    #plt.subplots_adjust(left=0.014, right=0.977, top=0.922, bottom=0.014)

    # print feature importance
    if classifier.__class__.__name__ == "RandomForestClassifier":
        print("feature importance:" )
        feature_names = ['length_ratio','mean_length','num_ttl','delta_num_ttl','dT','is_TCP']
        importances = list(classifier.feature_importances_)
        #print(importances)
        feat_imp = dict(zip(feature_names, classifier.feature_importances_))
        for feature in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True):
            print(feature)

    
def run_classification(data, labels): 
    model_error = [0, 0, 0, 0, 0]
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)    

    # Evaluate five standard classifiers. 
    classify(LR(), x_train, x_test, y_train, y_test)
    classify(KNeighborsClassifier(), x_train, x_test, y_train, y_test)
    classify(LinearSVC(), x_train, x_test, y_train, y_test)
    classify(DecisionTreeClassifier(), x_train, x_test, y_train, y_test)
    classify(RandomForestClassifier(), x_train, x_test, y_train, y_test)

run_classification(all_features.values, all_labels)