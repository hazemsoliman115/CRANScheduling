# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 20:11:25 2015

@author: hazem.soliman
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:27:41 2015

@author: hazem.soliman
"""
import numpy as np

class VarNode(object):
    
    def __init__(self, Thru, ReqNodes, NonReqNodes):
        self.Thru = Thru
        self.ReqNodes = ReqNodes
        self.NonReqNodes = NonReqNodes
        
    def UpdateThru(self):
        self.weight = sum(self.Thru)
        for i in range(len(self.Thru)):
            self.Thru[i] = self.weight/len(self.ReqNodes)
        try:
            self.Glob_Thru = self.weight/(len(self.NonReqNodes))
        except ZeroDivisionError:
            self.Glob_Thru = 0

#    def PropOtherNodes(self):
#        for i in range(len(self.NonReqNodes)):
#            self.Glob_Thru = self.weight/(len(self.NonReqNodes))
            
    
class FacNode(object):
    
    def __init__(self, connNodes, NonconnNodes):
        self. connNodes = connNodes
        self.NonconnNodes = NonconnNodes
        self.support = 0
        
        
if __name__ == "__main__":
    a = VarNode([0,0], None, None)
    b = VarNode([1,2], None, None)
    c = VarNode([1,1], None, None)
    d = VarNode([0, 0, 0], None, None)
    
    v_node = [a,b,c,d]
    
    f1 = FacNode([a,d], [b,c])
    f2 = FacNode([a,d], [b,c])
    f3 = FacNode([b,c,d], [a])
    f4 = FacNode([b,c], [a,d])
    
    f_node = [f1,f2,f3,f4]    
    
    a.ReqNodes = [f1,f2]
    a.NonReqNodes = [f3,f4]
    b.ReqNodes = [f3, f4]
    b.NonReqNodes = [f1,f2]
    c.ReqNodes = [f3,f4]
    c.NonReqNodes = [f1,f2]
    d.ReqNodes = [f1,f2,f3]
    d.NonReqNodes = [f4]
    
    a.UpdateThru()
    b.UpdateThru()
    c.UpdateThru()
    d.UpdateThru()
    
    for j in range(12):
        for k in f_node:
            k.support = sum([x.Glob_Thru for x in k.NonconnNodes])
            print(k.support)
        print('\n')
        print(a.Thru)
        print(a.Glob_Thru)
        print(b.Thru)
        print(b.Glob_Thru)
        print(c.Thru)
        print(c.Glob_Thru)
        print(d.Thru)
        print(d.Glob_Thru)
        print('\n')
        print('\n')
        temp_a_0 = a.Thru[0]
        temp_a_1 = a.Thru[1]
        
        temp_b_0 = b.Thru[0]
        temp_b_1 = b.Thru[1]
        
        temp_c_0 = c.Thru[0]
        temp_c_1 = c.Thru[1]
        
        temp_d_0 = d.Thru[0]
        temp_d_1 = d.Thru[1]
        temp_d_2 = d.Thru[2]
            
#        a.Thru[0] = min(50,max(temp_a_0-temp_d_0 + f1.support,-10))
#        a.Thru[1] = min(50,max(temp_a_1-temp_d_1 + f1.support,-10))
        a.Thru[0] = temp_a_0*(np.tanh(2.313*(temp_a_0-temp_d_0 + f1.support)))
        a.Thru[1] = temp_a_1*(np.tanh(2.313*(temp_a_1-temp_d_1 + f1.support)))
#        b.Thru[0] = min(50,max(temp_b_0-temp_c_0-temp_d_2 + f3.support,-10))
#        b.Thru[1] = min(50,max(temp_b_1-temp_c_1 + f4.support,-10))
        b.Thru[0] = temp_b_0*(np.tanh(2.313*(temp_b_0-temp_c_0-temp_d_2 + f3.support)))
        b.Thru[1] = temp_b_1*(np.tanh(2.313*(temp_b_1-temp_c_1 + f4.support)))
        
#        c.Thru[0] = min(50,max(temp_c_0-temp_b_0-temp_d_2 + f3.support,-10))
#        c.Thru[1] = min(50,max(temp_c_1-temp_b_1 + f4.support,-10))
        c.Thru[0] = temp_c_0*(np.tanh(2.313*(temp_c_0-temp_b_0-temp_d_2 + f3.support)))
        c.Thru[1] = temp_c_1*(np.tanh(2.313*(temp_c_1-temp_b_1 + f4.support)))
        
#        d.Thru[0] = min(50,max(temp_d_0-temp_a_0 + f1.support,-10))
#        d.Thru[1] = min(50,max(temp_d_1-temp_a_1 + f2.support,-10))
#        d.Thru[2] = min(50,max(temp_d_2-temp_b_0-temp_c_0 + f3.support,-10))
        d.Thru[0] = temp_d_0*(np.tanh(2.313*(temp_d_0-temp_a_0 + f1.support)))
        d.Thru[1] = temp_d_1*(np.tanh(2.313*(temp_d_1-temp_a_1 + f2.support)))
        d.Thru[2] = temp_d_2*(np.tanh(2.313*(temp_d_2-temp_b_0-temp_c_0 + f3.support)))

        
        a.UpdateThru()
        b.UpdateThru()
        c.UpdateThru()
        d.UpdateThru()
        
    print(a.Thru)
    print(b.Thru)
    print(c.Thru)
    print(d.Thru)
#    
#    a = np.matrix([[1, 0, 1, 0, 1, 0, -1, 0, 0, 0],[ 0, 1, 0, 1, 0, 1, 0, -1, 0, 0],[ 1, 0, 1, 0, -1, 0, 0, 0, -1, 0],[ 0, 1, 0, 1, 0, -1, 0, 0, 0, -1],[ 1, 0, -1, 0, 1, 0, 0, 0, -1, 0],[ 0, 1, 0, -1, 0, 1, 0, 0, 0, -1],[ -1, 0, 1, 0, 1, 0, 1, 0, 0, 0],[ 0, -1, 0, 1, 0, 1, 0, 1, 0, 0],[ 1, 0, -1, 0, -1, 0, 0, 0, 1, 0],[ 0, 1, 0, -1, 0, -1, 0, 0, 0, -1]])
#    b = np.array([[1, 0, 1, 0, 1, 0, -1, 0, 0, 0],[ 0, 1, 0, 1, 0, 1, 0, -1, 0, 0],[ 1, 0, 1, 0, -1, 0, 0, 0, -1, 0],[ 0, 1, 0, 1, 0, -1, 0, 0, 0, -1],[ 1, 0, -1, 0, 1, 0, 0, 0, -1, 0],[ 0, 1, 0, -1, 0, 1, 0, 0, 0, -1],[ -1, 0, 1, 0, 1, 0, 1, 0, 0, 0],[ 0, -1, 0, 1, 0, 1, 0, 1, 0, 0],[ 1, 0, -1, 0, -1, 0, 0, 0, 1, 0],[ 0, 1, 0, -1, 0, -1, 0, 0, 0, -1]])
#       
#    c = np.matrix([[1,1,1,-0.5],[1,1,-1,-0.5],[1,-1,1,-0.5],[-1,-1,-1,1]])
#    d = np.array([[1,1,1,-0.5],[1,1,-1,-0.5],[1,-1,1,-0.5],[-1,-1,-1,1]])
#    
#    print(c)    
#    print(np.transpose(c))
#    print(c*np.transpose(c))
#    print((c**15))
#    print(np.linalg.eigvals(d))