# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:12:24 2015

@author: hazem.soliman
"""

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
        self.Thru = sum(Thru)
        self.ReqNodes = ReqNodes
        self.NonReqNodes = NonReqNodes
        
    def UpdateThru(self):
        try:
            self.Glob_Thru = 0*self.Thru/(len(self.NonReqNodes))
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
    t_a = 0
    t_b = 0
    t_c = 2
    t_d = 13
    t_e = 5.5
    t_f = 0
    t_g = 5.5
    a = VarNode([t_a/2,t_a/2], None, None)
    b = VarNode([t_b/2,t_b/2], None, None)
    c = VarNode([t_c/2,t_c/2], None, None)
    d = VarNode([t_d/3,t_d/3,t_d/3], None, None)
    e = VarNode([t_e], None, None)
    f = VarNode([t_f], None, None)
    g = VarNode([t_g], None, None)
    
    
    v_node = [a,b,c,d]
    
    f1 = FacNode([a,d,e], [b,c,f,g])
    f2 = FacNode([a,d,f], [b,c,e,g])
    f3 = FacNode([b,c,d,g], [a,e,f])
    f4 = FacNode([b,c], [a,d,e,f,g])
    
    f_node = [f1,f2,f3,f4]    
    
    a.ReqNodes = [f1,f2]
    a.NonReqNodes = [f3,f4]
    b.ReqNodes = [f3, f4]
    b.NonReqNodes = [f1,f2]
    c.ReqNodes = [f3,f4]
    c.NonReqNodes = [f1,f2]
    d.ReqNodes = [f1,f2,f3]
    d.NonReqNodes = [f4]
    e.ReqNodes = [f1]
    e.NonReqNodes = [f2,f3,f4]
    f.ReqNodes = [f2]
    f.NonReqNodes = [f1,f3,f4]
    g.ReqNodes = [f3]
    g.NonReqNodes = [f1,f2,f4]
    
    a.UpdateThru()
    b.UpdateThru()
    c.UpdateThru()
    d.UpdateThru()
    e.UpdateThru()
    f.UpdateThru()
    g.UpdateThru()
    
    for j in range(90):
        for k in f_node:
            k.support = sum([x.Glob_Thru for x in k.NonconnNodes])
            
        print('\n')
        print(a.Thru)
        print(b.Thru)
        print(c.Thru)
        print(d.Thru)
        print(e.Thru)
        print(f.Thru)
        print(g.Thru)
        print('\n')
        print('\n')
        
        
        temp_a = a.Thru
        temp_b = b.Thru
        temp_c = c.Thru
        temp_d = d.Thru
        temp_e = e.Thru
        temp_f = f.Thru
        temp_g = g.Thru

            
#        a.Thru =  t_a if j>10 and j%10 == 1 else  t_a*(0.5+0.5*np.tanh((0.1*j+1)*(temp_a-max(temp_d,temp_e+temp_f) + f1.support + f2.support))) 
# 
#
#        b.Thru = t_b if j>10 and j%20 == 1 else t_b*(0.5+0.5*np.tanh((0.1*j+1)*(temp_b-max(temp_c,temp_d,temp_g) + f3.support + f4.support))) 
# 
#        
#        c.Thru = t_c if j>10 and j%30 == 1 else t_c*(0.5+0.5*np.tanh((0.1*j+1)*(temp_c-max(temp_b,temp_d,temp_g) + f3.support + f4.support))) 
# 
#        
#        d.Thru = t_d if j>10 and j%40 == 1 else t_d*(0.5+0.5*np.tanh((0.1*j+1)*(temp_d-max(temp_a,temp_e+temp_f)-max(temp_b,temp_c,temp_g) + f1.support + f2.support + f3.support))) 
# 
#        e.Thru = t_e if j>10 and j%50 == 1 else t_e*(0.5+0.5*np.tanh((0.1*j+1)*(temp_e-max(temp_a,temp_d) ))) 
#        
#        f.Thru = t_f if j>10 and j%60 == 1 else t_f*(0.5+0.5*np.tanh((0.1*j+1)*(temp_f-max(temp_a,temp_d) ))) 
#        
#        g.Thru = t_g if j>10 and j%70 == 1 else t_g*(0.5+0.5*np.tanh((0.1*j+1)*(temp_g-max(temp_b,temp_c,temp_d) ))) 
            
        a.Thru =  t_a if j>1000 and j%10 == 1 else  t_a*(0.5+0.5*np.tanh((0.1*j+1)*(temp_a-max(temp_d/3,temp_e)-max(temp_d/3,temp_f) + f1.support + f2.support))) 
 

        b.Thru = t_b if j>1000 and j%20 == 1 else t_b*(0.5+0.5*np.tanh((0.1*j+1)*(temp_b-max(temp_c/2,temp_d/3,temp_g)-max(temp_c/2,temp_d/3) + f3.support + f4.support))) 
 
        
        c.Thru = t_c if j>1000 and j%30 == 1 else t_c*(0.5+0.5*np.tanh((0.1*j+1)*(temp_c-max(temp_b/2,temp_d/3,temp_g)-max(temp_b/2,temp_d/3) + f3.support + f4.support))) 
 
        
        d.Thru = t_d if j>1000 and j%40 == 1 else t_d*(0.5+0.5*np.tanh((0.1*j+1)*(temp_d-max(temp_a/2,temp_e)-max(temp_a/2,temp_f)-max(temp_b/2,temp_c/2,temp_g) + f1.support + f2.support + f3.support))) 
 
        e.Thru = t_e if j>1000 and j%50 == 1 else t_e*(0.5+0.5*np.tanh((0.1*j+1)*(temp_e-max(temp_a/2,temp_d/3) ))) 
        
        f.Thru = t_f if j>1000 and j%60 == 1 else t_f*(0.5+0.5*np.tanh((0.1*j+1)*(temp_f-max(temp_a/2,temp_d/3) ))) 
        
        g.Thru = t_g if j>1000 and j%70 == 1 else t_g*(0.5+0.5*np.tanh((0.1*j+1)*(temp_g-max(temp_b/2,temp_c/2,temp_d/3) )))
        

        
        a.UpdateThru()
        b.UpdateThru()
        c.UpdateThru()
        d.UpdateThru()
        e.UpdateThru()
        f.UpdateThru()
        g.UpdateThru()
        
    print(a.Thru)
    print(b.Thru)
    print(c.Thru)
    print(d.Thru)
    print(e.Thru)
    print(f.Thru)
    print(g.Thru)
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