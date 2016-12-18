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
    Num_Flows_List = [5,10,15,20,25,30,35,40,45,50,55,60]#[5,10,15,20,25,30]
    #Num_Flows_List = [5]
    NumRes = 128
    
    error_vec = []
    error_vec_T = []
    for nflow_ind in range(len(Num_Flows_List)):
        List_Flows = []
        nflow = Num_Flows_List[nflow_ind]
        for i in range(nflow):
            x=Read_Chan_Trace(i+35+1)
            y=VarNode(i, x)
            List_Flows.append(y)
            
        chosen_flow = []
            
        for TTI in range(300):#range(len(List_Flows[0].channel_profile[0])):
            max_chan = -10
            select_flow = None
            for y in List_Flows:
                if y.channel_profile[0][TTI] >= max_chan:
                    max_chan = y.channel_profile[0][TTI]
                    select_flow = y
            chosen_flow.append(select_flow.iden)
            for y in List_Flows:
                if y is select_flow:
                    y.selected.append(1)
                else:
                    y.selected.append(0)

#    for y in List_Flows: 
#        if any(y.selected): 
#            X=[]               
#            X= np.reshape(y.channel_profile[0][0:300],(-1, 1)) 
#            clf = svm.SVC(kernel='rbf')               
#            clf.fit(X, y.selected)
#            
#            plt.figure(figsize=(10, 10))
#            plt.scatter(X, y.selected, c='g', marker = ".",label='data')
#            #plt.figure()
#            plt.scatter(X, clf.predict(X), c='b', marker = "p",label='Prediction')
            
            
        X=[]
        YX=[]
        for y in List_Flows:
            for i in range(300):#range(len(y.channel_profile[0])):
                X.append([y.channel_profile[0][i], y.channel_profile[0][i+1]])
                YX.append(y.selected[i])
        X = np.array(X)
        clf = svm.SVC(kernel='rbf', class_weight='auto')               
        clf.fit(X, YX)
        
        #plt.figure(figsize=(15, 15))
        #plt.scatter(X[:,0], X[:,1], c=YX, marker = "o",label='data')
        #plt.hold(True)
        #plt.figure(figsize=(10, 10))
        #plt.scatter(X[:,0]+2, X[:,1], c=clf.predict(X),  marker = "p",label='Prediction')
        
        d= []
        for i in range(len(YX)):
            if YX[i] == 1:
                d.append(clf.decision_function([X[i,0],X[i,1]]))
        #plt.figure(figsize=(10, 10))
        #plt.plot(d)
        print(0.5*sum(d+np.abs(d)))
        print(0.5*sum(d-np.abs(d)))
        
        v= []
        for i in range(len(YX)):
            if YX[i] == 0:
                v.append(clf.decision_function([X[i,0],X[i,1]]))
        #plt.figure(figsize=(10, 10))
        #plt.plot(v,'r')
        print(0.5*sum(v+np.abs(v)))
        print(0.5*sum(v-np.abs(v)))
        
        
    
#        plt.figure(figsize=(10, 10))
#        #plt.plot(chosen_flow)
#        for y in List_Flows:
#            #plt.plot(y.channel_profile[0][0:300])
#            plt.plot(y.selected)
#            plt.xlabel('Sub Frame Index')
#            plt.ylabel('Channel Gains and Selected users')
            #plt.legend([str(y.iden)], loc=8)
            #plt.savefig('ChannelandSelect_20082015.pdf', bbox_inches='tight')
            #plt.savefig('ChannelandSelect_20082015.eps', bbox_inches='tight')
            
        #plt.figure(figsize=(10, 10))   
        #plt.plot(YX,clf.predict(X))
        for y in List_Flows:
                y.selected = []
        for TTI in range(300):#range(len(List_Flows[0].channel_profile[0])):
            max_chan = -10
            select_flow = None
            for y in List_Flows:
                if y.channel_profile[40][TTI] >= max_chan:
                    max_chan = y.channel_profile[40][TTI]
                    select_flow = y
            for y in List_Flows:
                if y is select_flow:
                    y.selected.append(1)
                else:
                    y.selected.append(0)
        X_T=[]  
        YX_T=[]
        for y in List_Flows:
            for i in range(300):#range(len(y.channel_profile[0])):
                X_T.append([y.channel_profile[40][i], y.channel_profile[40][i+1]])
                YX_T.append(y.selected[i])
        X_T = np.array(X_T)
            
        error_vec.append(np.abs(sum(YX-clf.predict(X))/len(YX)))
        error_vec_T.append(np.abs(sum(YX_T-clf.predict(X_T))/len(YX_T)))
    plt.figure(figsize=(10, 10))
    plt.plot(Num_Flows_List,error_vec, label='Training Data')
    plt.plot(Num_Flows_List,error_vec_T,'r', label='Testing Data')
    plt.xlabel('Number of Flows')
    plt.legend()
    plt.ylabel('Predicition Error')
    #plt.savefig('SVMClassification_UnCorr_20082015.pdf', bbox_inches='tight')
    #plt.savefig('SVMClassification_UnCorr_20082015.eps', bbox_inches='tight')
    
            
       

            
                
            
                
           
            
            
        
    
#                
#       