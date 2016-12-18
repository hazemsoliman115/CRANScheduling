# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:32:43 2015

@author: hazem.soliman
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:46:29 2015

@author: hazem.soliman
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 18:50:49 2015

@author: hazem.soliman
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:21:43 2015

@author: hazem.soliman
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:21:32 2015

@author: hazem.soliman
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:26:41 2015

@author: hazem.soliman
"""
import random
import operator
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from cvxopt import matrix, solvers
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

def Read_Chan_Trace(j):
    with open('FadingTraces/Channel_Profile'+str(j)+'.csv', 'rt') as csvfile:
        chanreader = csv.reader(csvfile)
        chan_matrix = []
        for row in chanreader:
            for i in range(len(row)):
                row[i] = float(row[i])
            chan_matrix.append(row)
    return chan_matrix
    
class Resour(object):
    """ A class for the resource node used in the interval scheduling """
    def __init__(self, iden):
        """ constructor with attributes: identity(number), flows that start here and flows that end here """
        self.iden = iden
        self.starting = []
        self.ending = []

class VarNode(object):
    """ The main node for scheduling entities, each object represents a flow """
    def __init__(self, iden, channel_profile):
        """ constructor """
        self.iden = iden
        self.channel_profile = channel_profile
        self.selected = []
        self.throughput = [10]
        self.Res_size = len(self.channel_profile)
        self.original_Thru = None        
        self.Thru = None
        self.ReqNodes = None
        self.NonReqNodes = None
        self.ReqSize = None
        self.connVNodes = None
        self.NonconnVNodes = None
        self.psi = 0 # For interval scheduling
        
    
        
    
        

        
        
class WScheduler(object):
    """The main scheduler class, takes the form of a tree. Then users are sequentially fed to the tree and stored in the appropriate node depending upon their request. A method is used to examine the tree and return the list of scheduled users"""
    
    def __init__(self):
        """ Initialize the tree to an emoty tree """
        
        self.root = WSch_Node()
        
    
        
   
                    

            
        
                
    
   
        

        
        

        
        
if __name__ == "__main__":
    #Steps: 1. Generate nodes(flows) 2. Generate requests for nodes 3. Schedule using the two algorithms
    #Num_Flows =20
    Num_Flows_List = [10,20,30,40,50,60]#[5,10,15,20,25,30]
    #Num_Flows_List = [5]
    NumRes = 128
    clf_list = [svm.SVC(kernel='rbf', class_weight='auto', gamma=0, C=1), DecisionTreeClassifier(max_depth=5, class_weight='auto')]
    #clf_list = [LDA(), QDA(), GaussianNB(), linear_model.LogisticRegression(), KNeighborsClassifier(n_neighbors=3), DecisionTreeClassifier(max_depth=5, class_weight='auto'), SVC(class_weight='auto', kernel="linear", C=0.025), SVC(class_weight='auto', gamma=2, C=1)]
    clf_list_names = ['LDA', 'QDA', 'GaussianNB', 'Logisitic reg', 'K neigh', 'DTree', 'SVMLinear', 'SVMRBF']    
    error_vec_hit = [[] for i in range(len(clf_list))]
    error_vec_miss = [[] for i in range(len(clf_list))]
    error_vec = [[] for i in range(len(clf_list))]
    error_vec_T = [[] for i in range(len(clf_list))]
    error_vec_hit_T = [[] for i in range(len(clf_list))]
    error_vec_miss_T = [[] for i in range(len(clf_list))]
    throughput_mean = [[] for i in range(len(clf_list))]
    throughput_var = [[] for i in range(len(clf_list))]
    for clf_index in range(len(clf_list)):
        for nflow_ind in range(len(Num_Flows_List)):
            List_Flows = []
            nflow = Num_Flows_List[nflow_ind]
            for i in range(nflow):
                x=Read_Chan_Trace(i+35+1)
                y=VarNode(i, x)
                List_Flows.append(y)
                
            chosen_flow = []
                
            for TTI in range(600):#range(len(List_Flows[0].channel_profile[0])):
                max_weight = -10
                select_flow = None
                for y in List_Flows:
                    if y.channel_profile[0][TTI] >= max_weight and y.throughput[TTI] <=40:
                        max_weight = y.channel_profile[0][TTI]
                        select_flow = y
                if not select_flow:
                    select_flow = List_Flows[np.random.randint(1,Num_Flows_List[nflow_ind])]
                chosen_flow.append(select_flow.iden)
                for y in List_Flows:
                    if y is select_flow:
                        y.selected.append(1)
                        y.throughput.append(y.throughput[TTI] + y.channel_profile[0][TTI]) 
                    else:
                        y.selected.append(0)
                        y.throughput.append(y.throughput[TTI]) 
                
                
            X=[]
            YX=[]
            for y in List_Flows:
                for i in range(600):#range(len(y.channel_profile[0])):
                    X.append([y.channel_profile[0][i], y.throughput[i]])
                    YX.append(y.selected[i])
            X = np.array(X)
            #clf = svm.SVC(kernel='rbf', class_weight='auto')   # SVM 40% to 5%  
            #clf = QDA()             # Quadratic DA 20% to 5%
            #clf = GaussianNB()      # Gaussian 20% to 5%
            #clf = LDA()             # LDA 20% to 5%
            #clf = KNeighborsClassifier(n_neighbors=3)    # K neighbours, with 3 neighbours get 25% to 5% with nice curves where testing is worse than training
            #clf = DecisionTreeClassifier(max_depth=5)     # Decision Tree, 25% to 5% with nice figures  
            #clf = SVC(gamma=2, C=1)       # RBF SVM 18% to 2%
            #clf = SVC(kernel="linear", C=0.025)   # Linear kernel SVM, 20% to 2%
            #clf = linear_model.LogisticRegression()  # Logistic Regression, 20% to 2%
            clf = clf_list[clf_index]            
            clf.fit(X, YX)
            
#            for y in List_Flows:
#                plt.plot(y.throughput)
                #plt.plot(y.selected)
            
#            plt.figure(figsize=(10, 10))
#            plt.scatter(X[:,0], X[:,1], c=YX, marker = "o",label='data')
#            #plt.hold(True)
#            plt.figure(figsize=(10, 10))
#            plt.scatter(X[:,0]+2, X[:,1], c=clf.predict(X),  marker = "p",label='Prediction')
            
#            d= []
#            for i in range(len(YX)):
#                if YX[i] == 1:
#                    d.append(clf.decision_function([X[i,0],X[i,1]]))
#            #plt.figure(figsize=(10, 10))
#            #plt.plot(d)
#            print(0.5*sum(d+np.abs(d)))
#            print(0.5*sum(d-np.abs(d)))
#            
#            v= []
#            for i in range(len(YX)):
#                if YX[i] == 0:
#                    v.append(clf.decision_function([X[i,0],X[i,1]]))
#            #plt.figure(figsize=(10, 10))
#            #plt.plot(v,'r')
#            print(0.5*sum(v+np.abs(v)))
#            print(0.5*sum(v-np.abs(v)))
#            for i in range(len(YX)):
#                print(YX[i], clf.predict(X[i]))
#            plt.figure(figsize=(10, 10))   
#            plt.plot(YX,clf.predict(X),'*')
            for y in List_Flows:
                    y.selected = []
                    y.throughput = [10]
            for TTI in range(600):#range(len(List_Flows[0].channel_profile[0])):
                max_weight = -10
                select_flow = None
                for y in List_Flows:
                    if y.channel_profile[40][TTI] >= max_weight and y.throughput[TTI] <=40:
                        max_weight = y.channel_profile[40][TTI]
                        select_flow = y
                if not select_flow:
                    select_flow = List_Flows[np.random.randint(1,Num_Flows_List[nflow_ind])]
                chosen_flow.append(select_flow.iden)
                for y in List_Flows:
                    if y is select_flow:
                        y.selected.append(1)
                        y.throughput.append(y.throughput[TTI] + y.channel_profile[40][TTI]) 
                    else:
                        y.selected.append(0)
                        y.throughput.append(y.throughput[TTI]) 
            X_T=[]  
            YX_T=[]
            for y in List_Flows:
                for i in range(600):#range(len(y.channel_profile[0])):
                    X_T.append([y.channel_profile[40][i], y.throughput[i]])
                    YX_T.append(y.selected[i])
            X_T = np.array(X_T)
            z = clf.predict(X)
            z_T = clf.predict(X_T)
            e_h = 0
            l_h = 0
            e_m = 0
            l_m = 0
            for i in range(len(YX)):
                if YX[i] == 1:
                    e_h+=np.abs(YX[i]-z[i])
                    l_h+=1
                else:
                    e_m+=np.abs(YX[i]-z[i])
                    l_m+=1
            e_h_T = 0
            l_h_T = 0
            e_m_T = 0
            l_m_T = 0
            for i in range(len(YX_T)):
                if YX_T[i] == 1:
                    e_h_T+=np.abs(YX_T[i]-z_T[i])
                    l_h_T+=1
                else:
                    e_m_T+=np.abs(YX_T[i]-z_T[i])
                    l_m_T+=1
            error_vec_hit[clf_index].append(e_h/l_h)
            error_vec_miss[clf_index].append(e_m/l_m)
            error_vec_hit_T[clf_index].append(e_h_T/l_h_T)
            error_vec_miss_T[clf_index].append(e_m_T/l_m_T)
            error_vec[clf_index].append(sum(np.abs(YX-clf.predict(X))/len(YX)))
            error_vec_T[clf_index].append(sum(np.abs(YX_T-clf.predict(X_T))/len(YX_T)))
            throughput_Vec = []        
            for y in List_Flows:
                throughput_Vec.append(y.throughput[-1])
            throughput_mean[clf_index].append(np.mean(throughput_Vec))
            throughput_var[clf_index].append(np.var(throughput_Vec))
    plt.figure()
    marker_list = 'v^<>soxD'
    for clf_index in range(len(clf_list)):
        plt.plot(Num_Flows_List,error_vec[clf_index], label='Training Data'+str(clf_list_names[clf_index]))
        plt.plot(Num_Flows_List,error_vec_T[clf_index],marker = '*', label='Training Data'+str(clf_list_names[clf_index]))
        #plt.plot(Num_Flows_List,error_vec_T[clf_index], label=str(clf_list_names[clf_index]), marker=marker_list[clf_index])
        plt.xlabel('Number of Flows')
        plt.legend()
        plt.ylabel('Predicition Error Total')
        #plt.savefig('ClassifierComparison_21082015.pdf', bbox_inches='tight')
        #plt.savefig('ClassifierComparison_21082015.eps', bbox_inches='tight')
    plt.figure()
    marker_list = 'v^<>soxD'
    for clf_index in range(len(clf_list)):
        plt.plot(Num_Flows_List,error_vec_hit[clf_index], label='Training Data'+str(clf_list_names[clf_index]))
        plt.plot(Num_Flows_List,error_vec_hit_T[clf_index],marker = '*', label='Training Data'+str(clf_list_names[clf_index]))
        #plt.plot(Num_Flows_List,error_vec_T[clf_index], label=str(clf_list_names[clf_index]), marker=marker_list[clf_index])
        plt.xlabel('Number of Flows')
        plt.legend()
        plt.ylabel('Predicition Error Hit')
        #plt.savefig('ClassifierComparison_21082015.pdf', bbox_inches='tight')
        #plt.savefig('ClassifierComparison_21082015.eps', bbox_inches='tight')
    plt.figure()
    marker_list = 'v^<>soxD'
    for clf_index in range(len(clf_list)):
        plt.plot(Num_Flows_List,error_vec_miss[clf_index], label='Training Data'+str(clf_list_names[clf_index]))
        plt.plot(Num_Flows_List,error_vec_miss_T[clf_index],marker = '*', label='Training Data'+str(clf_list_names[clf_index]))
        #plt.plot(Num_Flows_List,error_vec_T[clf_index], label=str(clf_list_names[clf_index]), marker=marker_list[clf_index])
        plt.xlabel('Number of Flows')
        plt.legend()
        plt.ylabel('Predicition Error Miss')
        #plt.savefig('ClassifierComparison_21082015.pdf', bbox_inches='tight')
        #plt.savefig('ClassifierComparison_21082015.eps', bbox_inches='tight')
    plt.figure()
    marker_list = 'v^<>soxD'
    for clf_index in range(len(clf_list)):
        plt.plot(Num_Flows_List,throughput_mean[clf_index])
    plt.figure()
    marker_list = 'v^<>soxD'
    for clf_index in range(len(clf_list)):
        plt.plot(Num_Flows_List,throughput_var[clf_index])
#    plt.figure(figsize=(10, 10))    
#    for y in List_Flows:
#        y.metric = []
#        for i in range(600):
#            y.metric.append(y.channel_profile[40][i]/y.throughput[i])
#        plt.plot(y.metric)
##        
#    plt.figure(figsize=(10, 10))    
#    for y in List_Flows:
#        y.metric = []
#        for i in range(600):
#            y.metric.append(y.throughput[i])
#        plt.plot(y.metric)
#    
#    throughput_Vec = []        
#    for y in List_Flows:
#        throughput_Vec.append(y.throughput[-1])
#    print(np.mean(throughput_Vec))
#    print(np.var(throughput_Vec))
        
    
            
       

            
                
            
                
           
            
            
        
    
#                
#       