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

def Read_Chan_Trace(j):
    with open('FadingTraces/Channel_Profile'+str(j)+'.csv', 'rt') as csvfile:
        chanreader = csv.reader(csvfile)
        chan_matrix = []
        for row in chanreader:
            for i in range(len(row)):
                row[i] = float(row[i])
            chan_matrix.append(row)
    return chan_matrix

class VarNode(object):
    """ The main node for scheduling entities, each object represents a flow """
    def __init__(self, iden, channel_profile):
        """ constructor """
        self.iden = iden
        self.channel_profile = channel_profile
        self.Res_size = len(self.channel_profile)
        self.original_Thru = None        
        self.Thru = None
        self.ReqNodes = None
        self.NonReqNodes = None
        self.ReqSize = None
        self.connVNodes = None
        self.NonconnVNodes = None
        
    def GenResReq(self, NumRes, DepthTree, Allow_DepthTree, MaxDepthTree, TTI):
        """ (WFlow, int, int, int, int) -> NoneTyep Generate a reqsource request as follows: first, randomly select the requested resources size in a uniform fashion considering the depth of the tree to be the lower bound on the size. Second, choose the best set of resources of the chosen size within the allowed tree configuration considering the channel profile"""
        #Generate a request size by genearting a random integer between the maxdepthtree (from the number of global resources) and depthtree(as specified by the running envrionment). Then raise it as a power of 2
        self.req_size = 2**random.randint(DepthTree, MaxDepthTree)
        #Initialize an empty to store the weights of the various resource subsets to be examined
        self.req_weight=[]
        #Get the current channel profile
        col=[row[TTI] for row in self.channel_profile]
        #print(len(col))
        #print(type(col))
        #Get the weight for each subset
        for i in range(0,self.Res_size,max(self.req_size,2**Allow_DepthTree)):
            self.req_weight.append(sum(col[i:i+self.req_size]))
            
        #print(self.req_weight)
        #Find the subset of resources with the largest weight
        self.max_ResReq_index, self.max_ResReq = max(enumerate(self.req_weight), key = operator.itemgetter(1))
        self.Thru = self.max_ResReq
        self.original_Thru = self.max_ResReq
        #print(self.max_ResReq_index)
        #Encode the request, convert the resource subset index into binary, then fill it with zeros from the left according to its position in the tree
        self.ResReq = bin(self.max_ResReq_index)[2:].zfill(int(math.log2(len(self.req_weight))))
        
    def UpdateThru(self):
#        try:
            temp_NonconnVNodes = sorted(self.NonconnVNodes, key = lambda VarNode: len(VarNode.connVNodes), reverse=True)
            norm_Glob_Thru  = []
            for j in temp_NonconnVNodes:
                temp_sub_Thru  = [j.Thru]
                for k in j.connVNodes:
                    if k in temp_NonconnVNodes:
                        temp_sub_Thru.append(k.Thru)
                        temp_NonconnVNodes.remove(k)
                norm_Thru = max(temp_sub_Thru)
                norm_Glob_Thru.append(norm_Thru)
            self.Glob_Thru = sum(norm_Glob_Thru)


class FacNode(object):
    
    def __init__(self, connNodes, NonconnNodes):
        self. connNodes = connNodes
        self.NonconnNodes = NonconnNodes
        self.support = 0
        
        
class WScheduler(object):
    """The main scheduler class, takes the form of a tree. Then users are sequentially fed to the tree and stored in the appropriate node depending upon their request. A method is used to examine the tree and return the list of scheduled users"""
    
    def __init__(self):
        """ Initialize the tree to an emoty tree """
        
        self.root = WSch_Node()
        
    def CreateConflictGraph(self, List_Flows):
        """ A function that takes the list of flows and finds the connected and no-connected nodes for each flow """
        for fl in List_Flows:
            fl.connVNodes = []
            for c_fl in List_Flows:
                if c_fl is not  fl:
                    if c_fl.ResReq in fl.ResReq[0:len(c_fl.ResReq)] or fl.ResReq in c_fl.ResReq[0:len(fl.ResReq)]:
                        fl.connVNodes.append(c_fl)
                    
        for fl in List_Flows:
            fl.NonconnVNodes = [x for x in List_Flows if x not in fl.connVNodes]
        
        
    def insert_Flow(self, wflow_id = None, wflow_ResReq = None, cur_node = None):
        """ Insert a flow in the tree, initalizing any required along the path """
        #print(wflow_ResReq)
        if cur_node == None:
            cur_node = self.root
            if wflow_ResReq[0] == '0':
                if cur_node.left == None:
                    cur_node.left = WSch_Node()
                self.insert_Flow(wflow_id, wflow_ResReq[1:], cur_node = cur_node.left)
            elif wflow_ResReq[0] == '1':
                if cur_node.right == None:
                    cur_node.right = WSch_Node()
                self.insert_Flow(wflow_id, wflow_ResReq[1:], cur_node = cur_node.right)
        else:
            if len(wflow_ResReq) == 0:
                    cur_node.list_of_flows.append(wflow_id)
                    return
            else:
                if wflow_ResReq[0] == '0':
                    if cur_node.left == None:
                        cur_node.left = WSch_Node()
                    self.insert_Flow(wflow_id, wflow_ResReq[1:], cur_node = cur_node.left)
                elif wflow_ResReq[0] == '1':                
                    if cur_node.right == None:
                        cur_node.right = WSch_Node()
                    self.insert_Flow(wflow_id, wflow_ResReq[1:], cur_node = cur_node.right)
                    
                    
                    
    def Find_Sch_Flows(self):
        self.root = WScheduler._Find_Sch_Flow(self.root)
        
    def _Find_Sch_Flow(cur_node):
        left_weight = 0
        right_weight = 0
        if cur_node.left:
            cur_node.left = WScheduler._Find_Sch_Flow(cur_node.left)
            for i in range(len(cur_node.left.winner_flows)):
                left_weight += cur_node.left.winner_flows[i].max_ResReq
        if cur_node.right:
            cur_node.right = WScheduler._Find_Sch_Flow(cur_node.right)
            for i in range(len(cur_node.right.winner_flows)):
                right_weight += cur_node.right.winner_flows[i].max_ResReq
                
                
        if cur_node.list_of_flows:
            cur_node.list_of_flows.sort(key = lambda WFlow:WFlow.max_ResReq, reverse=True)
            temp_winner  =  cur_node.list_of_flows[0]
            temp_winner_weight = temp_winner.max_ResReq
            if temp_winner_weight >= (left_weight + right_weight):
                cur_node.winner_flows.append(temp_winner)
                return cur_node
            else:
                if cur_node.left and cur_node.right:
                    cur_node.winner_flows.extend(cur_node.left.winner_flows)
                    cur_node.winner_flows.extend(cur_node.right.winner_flows)
                    return cur_node
                elif cur_node.left:
                    cur_node.winner_flows.extend(cur_node.left.winner_flows)
                    return cur_node
                elif cur_node.right:
                    cur_node.winner_flows.extend(cur_node.right.winner_flows)
                    return cur_node
        else:
            if cur_node.left and cur_node.right:
                cur_node.winner_flows.extend(cur_node.left.winner_flows)
                cur_node.winner_flows.extend(cur_node.right.winner_flows)
                return cur_node
            elif cur_node.left:
                cur_node.winner_flows.extend(cur_node.left.winner_flows)
                return cur_node
            elif cur_node.right:
                cur_node.winner_flows.extend(cur_node.right.winner_flows)
                return cur_node
                
    def General_Find_Sch_Flows(self, List_Flows, gain, no_iter):
        """ The general scheduling algorithm """
        for fl in List_Flows:
            fl.UpdateThru()
        for j in range(no_iter):
            for fl in List_Flows:
                fl.Thru = fl.original_Thru*(gain + 0.5+0.5*np.tanh((0.1*j+1)*(fl.Thru+fl.Glob_Thru-max([x.Thru+x.Glob_Thru for x in fl.connVNodes] if len(fl.connVNodes) > 0 else [0]) )))
      
    def Generate_LP_Matrix(self, List_Flows):
        """A function that generates the linear program weight matrix and vector, where the matrix is such that for any edge, a row should have
        a one at each node location """
        A_ub = []
        c = []
        for fl in List_Flows:
            for c_fl in fl.connVNodes:
                row = [0 for i in range(len(List_Flows))]
                row[fl.iden] = 1
                row[c_fl.iden] = 1
                A_ub.append(row)
                
        for fl in List_Flows:
            c.append(-1*fl.max_ResReq)
            
        return(A_ub, c)
        
    def Generate_Adjacency_Matrix(self, List_Flows):
        """A function that finds the adjacency matrix of our conflict graph"""
        A_Adj = [[0 for i in range(len(List_Flows))] for i in range(len(List_Flows))]
        for fl in List_Flows:
            for c_fl in fl.connVNodes:
                A_Adj[fl.iden][c_fl.iden] = 1
                A_Adj[c_fl.iden][fl.iden] = 1
        return(A_Adj)
        
class WSch_Node(object):
    """ The node of the scheduling tree. It should have a list of the flows who requested it and their asoociated info. It should also point to its two children"""
    
    def __init__(self, left_child = None, right_child = None):
        """ Initialize the list to an empty one"""
        
        self.list_of_flows = []
        self.winner_flows = []
        self.left = left_child
        self.right = right_child
        
        
        

        
        
if __name__ == "__main__":
    #Steps: 1. Generate nodes(flows) 2. Generate requests for nodes 3. Schedule using the two algorithms
    #Num_Flows =20
    Num_Flows_List = [5,10,15,20,25,30]
    NumRes = 128
    MaxDepthTree = 6
    Allow_DepthTree = 0
    DepthTree = 0
    gain = 0
    no_iter_algorithm = 20
    no_iter_statistic = 100
    
    Throughput_Overall = [0 for k in range(len(Num_Flows_List))]
    Throughput_Optimal = [0 for k in range(len(Num_Flows_List))]
    Throughput_Heuristic = [0 for k in range(len(Num_Flows_List))]
    Throughput_LP = [0 for k in range(len(Num_Flows_List))]
    Throughput_Mean_Difference = [0 for k in range(len(Num_Flows_List))]
    Throughput_Var_Difference = [0 for k in range(len(Num_Flows_List))]
    
    for nflow in range(len(Num_Flows_List)):
        print(nflow)
        for TTI in range(no_iter_statistic):
            List_Flows = []
            for i in range(Num_Flows_List[nflow]):
                x=Read_Chan_Trace(i+1)
                y=VarNode(i, x)
                y.GenResReq(NumRes, DepthTree, Allow_DepthTree, MaxDepthTree, TTI)
                #y.GenResReq_TreeConstraint(NumRes, DepthTree, Allow_DepthTree, MaxDepthTree, TTI)
                List_Flows.append(y)
                
            
                
            the_sched = WScheduler()   
            the_sched.CreateConflictGraph(List_Flows)
            
#            for fl in List_Flows:
#                print("Identity: ",fl.iden)
#                print("Request: ",fl.ResReq)
#                print("Output Optimal: ",fl.max_ResReq if fl in the_sched.root.winner_flows else 0)
#                print("Output Heueristic: ",fl.Thru-gain*fl.original_Thru)
#                print("Connected Nodes: ",[z.iden for z in fl.connVNodes])
#                print("Non-connected Nodes: ", [z.iden for z in fl.NonconnVNodes])
            for i in range(len(List_Flows)):
                the_sched.insert_Flow(List_Flows[i], List_Flows[i].ResReq)
            the_sched.Find_Sch_Flows()
            temp_thru = 0
            for i in range(len(the_sched.root.winner_flows)):
                temp_thru +=the_sched.root.winner_flows[i].max_ResReq
                
            temp_thru_2 = 0
            the_sched.General_Find_Sch_Flows(List_Flows, gain, no_iter_algorithm)
            for i in range(len(List_Flows)):
                temp_thru_2 += (List_Flows[i].Thru-gain*List_Flows[i].original_Thru)
            #print("Temp Throughput = ", temp_thru)
            #print("Temp Throughput = ", temp_thru_2)
            Throughput_Optimal[nflow] += temp_thru
            Throughput_Heuristic[nflow] += temp_thru_2
            Throughput_Mean_Difference[nflow] += temp_thru - temp_thru_2
            Throughput_Var_Difference[nflow] += 100*(temp_thru - temp_thru_2)/temp_thru
            
            A_Adj = the_sched.Generate_Adjacency_Matrix(List_Flows)
            
            A_ub,c = the_sched.Generate_LP_Matrix(List_Flows)
#            for fl in List_Flows:
#                print("Identity: ",fl.iden)
##                print("Request: ",fl.ResReq)
##                print("Output Optimal: ",fl.max_ResReq if fl in the_sched.root.winner_flows else 0)
##                print("Output Heueristic: ",fl.Thru-gain*fl.original_Thru)
#                print("Connected Nodes: ",[z.iden for z in fl.connVNodes])
##                print("Non-connected Nodes: ", [z.iden for z in fl.NonconnVNodes])
#            print(A_ub)
            b = [1 for k in range(len(A_ub))] if len(A_ub) > 0 else None
            bounds = tuple((0,1) for i in range(len(c)))
            res = optimize.linprog(c=c, A_ub=A_ub if len(A_ub) > 0 else None, b_ub=b, A_eq=None, b_eq=None, bounds=bounds, method='simplex', callback=None, options=None)
            #print(res.x)
            Flows_indicator = np.round(res.x)
            temp_thru_3 = 0
            temp_thru_4 = 0
            for i in range(len(List_Flows)):
                temp_thru_3 += Flows_indicator[i]*List_Flows[i].max_ResReq
            for i in range(len(List_Flows)):
                temp_thru_4 += List_Flows[i].max_ResReq
            Throughput_LP[nflow] += temp_thru_3
            Throughput_Overall[nflow] += temp_thru_4
            
        Throughput_Overall[nflow] = Throughput_Overall[nflow]/no_iter_statistic
        Throughput_Optimal[nflow] = Throughput_Optimal[nflow]/no_iter_statistic
        Throughput_Heuristic[nflow] = Throughput_Heuristic[nflow]/no_iter_statistic
        Throughput_LP[nflow] = Throughput_LP[nflow]/no_iter_statistic
        Throughput_Mean_Difference[nflow] = Throughput_Mean_Difference[nflow]/no_iter_statistic
        Throughput_Var_Difference[nflow] = Throughput_Var_Difference[nflow]/no_iter_statistic
        
    plt.figure(0)
    plt.plot(Num_Flows_List,Throughput_Optimal, marker  = '>')
    plt.plot(Num_Flows_List,Throughput_Heuristic, marker  = 'o')
    plt.plot(Num_Flows_List,Throughput_LP, marker  = '.')
    #plt.plot(Num_Flows_List,Throughput_Overall)
    plt.xlabel('Number of Flows')
    plt.ylabel('Throughput')
    plt.title('Throughput Comparison for Optimal and Heuristic Algorithm')
    plt.legend(['Optimal','Proposed Heuristic','Linear Program'], loc=0)
    #plt.savefig('ThroughputOptHeuristLP31102015.pdf', bbox_inches='tight')
    #plt.savefig('ThroughputOptHeuristLP31102015.eps', bbox_inches='tight')
    plt.figure(1)
    plt.plot(Num_Flows_List,Throughput_Mean_Difference, marker = 'v')
    plt.xlabel('Number of Flows')
    plt.ylabel('Throughput Loss')
    plt.title('Throughput Loss for the Heuristic Algorithm')
    #plt.savefig('ThroughputLoss31102015.pdf', bbox_inches='tight')
    #plt.savefig('ThroughputLoss31102015.eps', bbox_inches='tight')
    plt.figure(2)
    plt.plot(Num_Flows_List,Throughput_Var_Difference, marker = 'v')
    plt.xlabel('Number of Flows')
    plt.ylabel('Percentage of Throughput Loss')
    plt.title('Percentage of Throughput Loss for the Heuristic Algorithm')
    #plt.savefig('PercentThroughputLoss31102015.pdf', bbox_inches='tight')
    #plt.savefig('PercentThroughputLoss31102015.eps', bbox_inches='tight')
#                
#        #for i in range(len(List_Flows)):
#            #print(List_Flows[i])
#            #print(List_Flows[i].ResReq)
#            
#            the_sched.Find_Sch_Flows()
#
#        #for i in range(len(List_Flows)):
#            #print(List_Flows[i].ResReq)
#            #print(List_Flows[i].max_ResReq)
#    
#
#            for i in range(len(the_sched.root.winner_flows)):
#                temp_thru +=the_sched.root.winner_flows[i].max_ResReq
#                #print(the_sched.root.winner_flows[i].ResReq)
#                #print(the_sched.root.winner_flows[i].max_ResReq)
#            temp_No_Sch_Users += len(the_sched.root.winner_flows)
    
    
#    x=Read_Chan_Trace(4)
#    plt.plot(x[10])
#    plt.xlabel('SubFrame Number')
#    plt.ylabel('Power Gain')
#    plt.savefig('CHannelProfile.eps', bbox_inches='tight')
#    plt.savefig('CHannelProfile.pdf', bbox_inches='tight')