# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 18:41:54 2015

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
    with open('FadingTraces/Channel_Profile_1000'+str(j)+'.csv', 'rt') as csvfile:
        chanreader = csv.reader(csvfile)
        chan_matrix = []
        for row in chanreader:
            for i in range(len(row)):
                row[i] = float(row[i])
            chan_matrix.append(row)
    return chan_matrix

class Flow(object):
    def __init__(self, iden, channel, Num_PRBs, Num_TTIs):
        self.iden = iden
        self.channel = channel
        self.selected = [[] for PRB_Indx in range(Num_PRBs)]
        self.predicted = [[] for PRB_Indx in range(Num_PRBs)]
        self.throughput = [10]
        self.pred_throughput = [10]
        self.request = [[] for TTI in range(Num_TTIs)]
        self.predictor = None
        
    def Update_Throughput(self, TTI, PRB_Indx, chan_offset):
        if y.selected[PRB_Indx][TTI] == 1:
            y.throughput.append(y.throughput[TTI] + y.channel[PRB_Indx+chan_offset][TTI])
        else:
            y.throughput.append(y.throughput[TTI])
            
    def Update_Predict_Throughput(self, TTI, PRB_Indx, chan_offset):
        if y.predicted[PRB_Indx][TTI] == 1:
            y.pred_throughput.append(y.pred_throughput[TTI] + y.channel[PRB_Indx+chan_offset][TTI])
        else:
            y.pred_throughput.append(y.pred_throughput[TTI])
    
    def Sch_Predict(self, List_PRBs, TTI, PRB_Indx_Test, Scheduler_Type, chan_offset):
        if Scheduler_Type == 'PF':
            temp_dec = self.predictor.predict([self.channel[PRB_Indx_Test+chan_offset][TTI], self.pred_throughput[TTI]])
            if temp_dec == 1:
                self.predicted[PRB_Indx_Test].append(1)
            else:
                self.predicted[PRB_Indx_Test].append(0)
            if temp_dec == 1:
                self.request[TTI].append(List_PRBs[PRB_Indx_Test])
        elif Scheduler_Type == 'MW':
            temp_dec = self.predictor.predict(self.channel[PRB_Indx_Test+chan_offset][TTI])
            if temp_dec == 1:
                self.predicted[PRB_Indx_Test].append(1)
            else:
                self.predicted[PRB_Indx_Test].append(0)
            if temp_dec == 1:
                self.request[TTI].append(List_PRBs[PRB_Indx_Test])
        elif Scheduler_Type == 'Fairness':
            temp_dec = self.predictor.predict([self.channel[PRB_Indx_Test+chan_offset][TTI], self.pred_throughput[TTI]])
            if temp_dec == 1:
                self.predicted[PRB_Indx_Test].append(1)
            else:
                self.predicted[PRB_Indx_Test].append(0)
            if temp_dec == 1:
                self.request[TTI].append(List_PRBs[PRB_Indx_Test])
        
        
        
class Scheduler(object):
    def __init__(self):
        self.list_Sch_Flows = []
        
        
        
class PF_Scheduler(Scheduler):
    def __init__(self):
        super().__init__() 
            
        
    def Schedule(self, List_Flows, TTI, PRB_Indx, chan_offset):
        max_weight = -10
        select_flow = None
        for y in List_Flows:
            if y.channel[PRB_Indx+chan_offset][TTI]/y.throughput[TTI] >= max_weight:
                max_weight = y.channel[PRB_Indx+chan_offset][TTI]/y.throughput[TTI]
                select_flow = y
        for y in List_Flows:
            if y is select_flow:
                y.selected[PRB_Indx].append(1)
            else:
                y.selected[PRB_Indx].append(0)
        return(select_flow)
        
        
class MW_Scheduler(Scheduler):
    def __init__(self):
        super().__init__()
        
    def Schedule(self, List_Flows, TTI, PRB_Indx, chan_offset):
        max_weight = -10
        select_flow = None
        for y in List_Flows:
            if y.channel[PRB_Indx+chan_offset][TTI] >= max_weight:
                max_weight = y.channel[PRB_Indx+chan_offset][TTI]
                select_flow = y
        for y in List_Flows:
            if y is select_flow:
                y.selected[PRB_Indx].append(1)
            else:
                y.selected[PRB_Indx].append(0)
        return(select_flow)
        
class Fairness_Scheduler(Scheduler):
    def __init__(self):
        super().__init__()
        
    def Schedule(self, List_Flows, TTI, PRB_Indx, w_Gamma, chan_offset):
        max_weight = -1*float("inf")
        select_flow = None
        for y in List_Flows:
            if y.channel[PRB_Indx+chan_offset][TTI]-w_Gamma*(y.channel[PRB_Indx+chan_offset][TTI]+y.channel[PRB_Indx+chan_offset][TTI]/len(List_Flows))**2-2*w_Gamma*(y.channel[PRB_Indx+chan_offset][TTI]+y.channel[PRB_Indx+chan_offset][TTI]/len(List_Flows))*(y.throughput[-1] - sum([zw.throughput[-1] for zw in List_Flows])/len(List_Flows)) >= max_weight:
                max_weight = y.channel[PRB_Indx+chan_offset][TTI]-w_Gamma*(y.channel[PRB_Indx+chan_offset][TTI]+y.channel[PRB_Indx+chan_offset][TTI]/len(List_Flows))**2-2*w_Gamma*(y.channel[PRB_Indx+chan_offset][TTI]+y.channel[PRB_Indx+chan_offset][TTI]/len(List_Flows))*(y.throughput[-1] - sum([zw.throughput[-1] for zw in List_Flows])/len(List_Flows))
                select_flow = y
        for y in List_Flows:
            if y is select_flow:
                y.selected[PRB_Indx].append(1)
            else:
                y.selected[PRB_Indx].append(0)
        return(select_flow)
        
class Predictor(object):
    def __init__(self, clf):
        self.X_DATA = []
        self.X_DATA_Test = []
        self.Y_DECISION = []
        self.Y_DECISION_Test = []
        self.clf = clf
        self.error_hit = []
        self.error_miss = []
        self.error = []
        
    def insert_data(self, List_Flows, TTI, PRB_Indx, Scheduler_Type, chan_offset):
        if Scheduler_Type == 'PF':
            for y in List_Flows:
                self.X_DATA.append([y.channel[PRB_Indx+chan_offset][TTI], y.throughput[TTI]])
                self.Y_DECISION.append(y.selected[PRB_Indx][TTI])
        elif Scheduler_Type == 'MW':
            for y in List_Flows:
                self.X_DATA.append([y.channel[PRB_Indx+chan_offset][TTI]])
                self.Y_DECISION.append(y.selected[PRB_Indx][TTI])
        elif Scheduler_Type == 'Fairness':
            for y in List_Flows:
                self.X_DATA.append([y.channel[PRB_Indx+chan_offset][TTI], y.throughput[TTI]])
                self.Y_DECISION.append(y.selected[PRB_Indx][TTI])
                
                
    def insert_data_Test(self, List_Flows, TTI, PRB_Indx, Scheduler_Type, chan_offset):
        if Scheduler_Type == 'PF':
            for y in List_Flows:
                self.X_DATA_Test.append([y.channel[PRB_Indx+chan_offset][TTI], y.throughput[TTI]])
                self.Y_DECISION_Test.append(y.selected[PRB_Indx][TTI])
        elif Scheduler_Type == 'MW':
            for y in List_Flows:
                self.X_DATA_Test.append([y.channel[PRB_Indx+chan_offset][TTI]])
                self.Y_DECISION_Test.append(y.selected[PRB_Indx][TTI])
        elif Scheduler_Type == 'Fairness':
            for y in List_Flows:
                self.X_DATA_Test.append([y.channel[PRB_Indx+chan_offset][TTI], y.throughput[TTI]])
                self.Y_DECISION_Test.append(y.selected[PRB_Indx][TTI])
            
    def predict(self):
        #print(len(self.X_DATA))
        self.X_DATA = np.array(self.X_DATA)
        #print(len(self.X_DATA))
        #self.X_DATA.shape = (len(self.X_DATA),1)
        self.clf.fit(self.X_DATA, self.Y_DECISION)
        return(self.clf)
        
    def Find_efficiency(self):
        self.error = sum(np.abs(self.Y_DECISION_Test-self.clf.predict(self.X_DATA_Test))/len(self.Y_DECISION_Test))
        e_h=0
        l_h=0
        e_m=0
        l_m=0
        for i in range(len(self.Y_DECISION_Test)):
                if self.Y_DECISION_Test[i] == 1:
                    e_h+=np.abs(self.Y_DECISION_Test[i]-self.clf.predict(self.X_DATA_Test[i]))
                    l_h+=1
                else:
                    e_m+=np.abs(self.Y_DECISION_Test[i]-self.clf.predict(self.X_DATA_Test[i]))
                    l_m+=1
        self.error_hit=e_h/l_h
        print(e_h/l_h)
        self.error_miss=e_m/l_m
        print(e_m/l_m)
        return(self.error,self.error_hit,self.error_miss)
        
        
class PRB(object):
    def __init__(self, iden):
        self.iden = iden
        self.buffer = []
        self.scheduled = None
        self.selected = None
        
if __name__ == "__main__":
    # Initialization: Number of flows, number of PRBs, list of predictors
    Num_Flows_List = [10,20,30,40,50,60]
    Num_PRBs = 1
    Num_TTIs = 800
    
    #w_Gamma_list = [0, 0.01, 0.05, 0.09, 0.1, 0.3, 0.5, 0.7, 0.9]
    w_Gamma_list = [0]
    Scheduler_Type = 'MW' #'MW' 'PF' 'Fairness'
    Conflict_Resolver = 'Random' # 'Random'  'Buffering'
    #clf_list = [svm.SVC(kernel='rbf', class_weight='auto', gamma=0, C=1), DecisionTreeClassifier(max_depth=5, class_weight={1:19})] # use for PF loss = 10.87%
    clf_list = [svm.SVC(kernel='rbf', class_weight='auto', gamma=0, C=1), DecisionTreeClassifier(max_depth=5, class_weight={1: 70})] #use for MW loss = 14%
    #clf_list = [svm.SVC(kernel='rbf', class_weight='auto', gamma=0, C=1)]
    clf_list_names = ['SVM-RBF','Decision Tree','GaussianNB']    
    # Initalization of Empty Vectors for Results
    error_vec_hit = [[] for i in range(len(clf_list))]
    error_vec_miss = [[] for i in range(len(clf_list))]
    error_vec = [[] for i in range(len(clf_list))]
    error_vec_T = [[] for i in range(len(clf_list))]
    error_vec_hit_T = [[] for i in range(len(clf_list))]
    error_vec_miss_T = [[] for i in range(len(clf_list))]
    # Assume for now infinitely backlogged buffers, otherwise for each user
    # equip with a buffer and an arrival process, generate arrival times at the
    # beginning, at each TTI check if there is new arrival and extend the buffer
    Throughput_Vector_Sch = [[0 for j in range(len(Num_Flows_List))] for i in range(len(clf_list))]
    Throughput_Vector_Predict = [[0 for j in range(len(Num_Flows_List))] for i in range(len(clf_list))]
    err = [[[0 for num_flow_ind in range(len(Num_Flows_List))] for clf_ind in range(len(clf_list))] for j in range(len(w_Gamma_list))]
    err_h = [[[0 for num_flow_ind in range(len(Num_Flows_List))] for clf_ind in range(len(clf_list))] for j in range(len(w_Gamma_list))]
    err_m = [[[0 for num_flow_ind in range(len(Num_Flows_List))] for clf_ind in range(len(clf_list))] for j in range(len(w_Gamma_list))]
    Thru_Ratio = [[0 for num_flow_ind in range(len(Num_Flows_List))] for j in range(len(w_Gamma_list))]
    for w_Gamma_ind in range(len(w_Gamma_list)):
        # Loop over predictors
        for clf_ind in range(len(clf_list)):
            # Loop over number of flows
            for num_flow_ind in range(len(Num_Flows_List)):
                # Create list of users and initialize their channels
                List_Flows = []
                nflow = Num_Flows_List[num_flow_ind]
                for i in range(nflow):
                    x=Read_Chan_Trace(i+1)
                    y=Flow(i, x, Num_PRBs, Num_TTIs)
                    List_Flows.append(y)
                                
                # Create PRB list    
                PRB_list = [[] for TTI in range(Num_TTIs)]
                
                # Learning Stuff
                C_Predictor = Predictor(clf_list[clf_ind])
                # Loop over TTIs
                chan_offset = 0
                for TTI in range(Num_TTIs):
                    # Schedule, feed selected user into the buffer
                    # loop over PRBs
                    for PRB_Indx in range(Num_PRBs):
                        PRB_list[TTI].append(PRB(PRB_Indx))
                        if Scheduler_Type == 'PF':
                            Scheduler = PF_Scheduler()                       
                        elif Scheduler_Type == 'MW':
                            Scheduler = MW_Scheduler()
                        elif Scheduler_Type == 'Fairness':
                            Scheduler = Fairness_Scheduler()
                            
                        if Scheduler_Type == 'Fairness':
                            PRB_list[TTI][PRB_Indx].scheduled = Scheduler.Schedule(List_Flows, TTI, PRB_Indx, w_Gamma_list[w_Gamma_ind], chan_offset)
                        else:
                            PRB_list[TTI][PRB_Indx].scheduled = Scheduler.Schedule(List_Flows, TTI, PRB_Indx, chan_offset)
                        
                        # Learn Scheduling Decision
                        C_Predictor.insert_data(List_Flows, TTI, PRB_Indx, Scheduler_Type, chan_offset)
                        
                        # Update Throughput
                        for y in List_Flows:
                            y.Update_Throughput(TTI, PRB_Indx, chan_offset)

                    
                    
                            
                        
                
    
                for y in List_Flows:
                    y.selected[0] = []
                # Feed predictor into users
                temp_predictor = C_Predictor.predict()
                for y in List_Flows:
                    y.predictor = temp_predictor
            
                PRB_list_Test = [[] for TTI in range(Num_TTIs)]
                # Loop over TTIs
                chan_offset = 40
                for TTI in range(Num_TTIs):
                    for PRB_Indx_Test in range(Num_PRBs):

                        # Schedule, feed selected user into the buffer
                        # loop over PRBs
                        PRB_list_Test[TTI].append(PRB(PRB_Indx_Test))
                        if Scheduler_Type == 'PF':
                            Scheduler = PF_Scheduler()                       
                        elif Scheduler_Type == 'MW':
                            Scheduler = MW_Scheduler()
                        elif Scheduler_Type == 'Fairness':
                            Scheduler = Fairness_Scheduler()
                            
                        
                        if Scheduler_Type == 'Fairness':
                            PRB_list_Test[TTI][PRB_Indx_Test].scheduled = Scheduler.Schedule(List_Flows, TTI, PRB_Indx_Test, w_Gamma_list[w_Gamma_ind], chan_offset)
                        else:
                            PRB_list_Test[TTI][PRB_Indx_Test].scheduled = Scheduler.Schedule(List_Flows, TTI, PRB_Indx_Test, chan_offset)
                            
                        C_Predictor.insert_data_Test(List_Flows, TTI, PRB_Indx_Test, Scheduler_Type, chan_offset)
                        
                        # Predict decisions
                        for y in List_Flows:
                            y.Sch_Predict(PRB_list_Test, TTI, PRB_Indx_Test, Scheduler_Type, chan_offset)
                            if PRB_list_Test[PRB_Indx_Test] in y.request[TTI]:
                                PRB_list_Test[TTI][PRB_Indx_Test].buffer.append(y)
                        
                    # In case of conflict:
                        # 1. Choose at Random, criteria Throughput & fairness
                        # 2. Buffer, criteria Buffer Size, throughput & fairness
                        Throughput_Vector_Sch[clf_ind][num_flow_ind]+=PRB_list_Test[TTI][PRB_Indx_Test].scheduled.channel[PRB_Indx_Test+chan_offset][TTI]
                        if Conflict_Resolver == 'Random':
                            Throughput_Vector_Predict[clf_ind][num_flow_ind]+=PRB_list_Test[TTI][PRB_Indx_Test].buffer[np.random.randint(0,len(PRB_list_Test[TTI][PRB_Indx_Test].buffer))].channel[PRB_Indx_Test+chan_offset][TTI] if len(PRB_list_Test[TTI][PRB_Indx_Test].buffer)>0 else 0
    
                        # Update Throughput
                        for y in List_Flows:
                            y.Update_Throughput(TTI, PRB_Indx_Test, chan_offset)
                            y.Update_Predict_Throughput(TTI, PRB_Indx_Test, chan_offset)

                [err[w_Gamma_ind][clf_ind][num_flow_ind], err_h[w_Gamma_ind][clf_ind][num_flow_ind], err_m[w_Gamma_ind][clf_ind][num_flow_ind]] = C_Predictor.Find_efficiency()  
        Thru_Ratio[w_Gamma_ind] = (-Throughput_Vector_Predict[0][-1]+Throughput_Vector_Sch[0][-1])/Throughput_Vector_Sch[0][-1]
    marker_list='o>sdv^'
    plt.figure(0)
    for clf_ind in range(len(clf_list)):
        if clf_ind == 0:
            plt.plot(Num_Flows_List,[xy/Num_TTIs for xy in Throughput_Vector_Sch[clf_ind]], label = 'Centralized', marker = '*')
        plt.plot(Num_Flows_List,[xy/Num_TTIs for xy in Throughput_Vector_Predict[clf_ind]], label = clf_list_names[clf_ind], marker = marker_list[clf_ind])
        plt.legend(loc=2)
        plt.xlabel('Number of Users')
        plt.ylabel('Expected SINR')
        plt.title('Expected SINR versus Number of Users')
        #plt.savefig('PFThroughput23122015.pdf', bbox_inches='tight')
        #plt.savefig('PFThroughput23122015.eps', bbox_inches='tight')
        
    plt.figure(1)
    cmap=plt.get_cmap('jet')
    for clf_ind in range(len(clf_list)):
        plt.plot(Num_Flows_List, err[0][clf_ind], label = 'Total Error+'+clf_list_names[clf_ind], marker = marker_list[3*clf_ind+0])
        plt.plot(Num_Flows_List, err_h[0][clf_ind], label = 'Hit Error+'+clf_list_names[clf_ind], marker = marker_list[3*clf_ind+1])
        plt.plot(Num_Flows_List, err_m[0][clf_ind], label = 'Miss Error+'+clf_list_names[clf_ind], marker = marker_list[3*clf_ind+2])
        plt.legend(bbox_to_anchor=(1, 1.3),borderaxespad=0,ncol=2)
        plt.xlabel('Number of Users')
        plt.ylabel('Prediction Errors')
        #plt.savefig('PFAccuracy23122015.pdf', bbox_inches='tight')
        #plt.savefig('PFAccuracy23122015.eps', bbox_inches='tight')
        
#    Figures for the Fairness Scheduler
    plt.figure(0)
    plt.plot(w_Gamma_list,[x[0][-1] for x in err], marker = '*', label = 'Total Error')
    plt.xlabel('Beta')
    plt.ylabel('Prediction Errors')
    plt.title('Prediction Errors Versus Beta')
    plt.hold(True)
    plt.plot(w_Gamma_list,[x[0][-1] for x in err_h], marker = 'o', label = 'Hit Error')
    plt.hold(True)
    plt.plot(w_Gamma_list,[x[0][-1] for x in err_m], marker = '>',label = 'Miss Error')
    plt.legend(loc = 7)
    #plt.savefig('FairnessvsBeta13092015.pdf', bbox_inches='tight')
    #plt.savefig('FairnessvsBeta13092015.eps', bbox_inches='tight')
    plt.figure(1)
    plt.plot(w_Gamma_list,[xx*100 for xx in Thru_Ratio],marker = 'p')
    plt.xlabel('Beta')
    plt.ylabel('Performance Loss %')
    plt.title('Prediction Errors Versus Beta')
    #plt.savefig('ThrouLossvsBeta13092015.pdf', bbox_inches='tight')
    #plt.savefig('ThrouLossvsBeta13092015.eps', bbox_inches='tight')
#    for TTI in range(Num_TTIs):
#        print(PRB_list_Test[TTI][PRB_Indx_Test].scheduled.channel[0][TTI])
#        print([x.channel[0][TTI] for x in PRB_list_Test[TTI][PRB_Indx_Test].buffer])
#        print([C_Predictor.clf.predict(z.channel[0][TTI]) for z in PRB_list_Test[TTI][PRB_Indx_Test].buffer])
    u_error = 0
    for y in List_Flows:
        for i in range(Num_TTIs):
            u_error += np.abs(y.selected[PRB_Indx][i]-y.predicted[PRB_Indx][i])