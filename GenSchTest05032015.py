# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:28:05 2015

@author: hazem.soliman
"""

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
import matplotlib.pyplot as plt

class VarNode(object):
    
    def __init__(self, Thru, ReqNodes, NonReqNodes):
        self.Thru = sum(Thru)
        self.ReqNodes = ReqNodes
        self.NonReqNodes = NonReqNodes
        self.connVNodes = None
        self.NonconnVNodes = None
        
    def UpdateThru(self):
        try:
            self.Glob_Thru = max([sum([x.Thru if x not in y.connVNodes else 0 for x in self.NonconnVNodes]) for y in self.NonconnVNodes])
        except:
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
    t_a = 3
    t_b = 3
    t_c = 2
    t_d = 5.5
    t_e = 1
    t_f = 1
    t_g = 0
    gain = 1
    a = VarNode([t_a/2,t_a/2], None, None)
    b = VarNode([t_b/2,t_b/2], None, None)
    c = VarNode([t_c/2,t_c/2], None, None)
    d = VarNode([t_d/3,t_d/3,t_d/3], None, None)
    e = VarNode([t_e], None, None)
    f = VarNode([t_f], None, None)
    g = VarNode([t_g], None, None)
    
    a.connVNodes = [d,e,f]
    a.NonconnVNodes = [b,c,g]
    b.connVNodes = [c,d,g]
    b.NonconnVNodes = [a,e,f]
    c.connVNodes = [b,d,g]
    c.NonconnVNodes = [a,e,f]
    d.connVNodes = [a,b,c,e,f,g]
    d.NonconnVNodes = []
    e.connVNodes = [a,d]
    e.NonconnVNodes = [b,c,f,g]
    f.connVNodes = [a,d]
    f.NonconnVNodes = [b,c,e,g]
    g.connVNodes = [b,c,d]
    g.NonconnVNodes = [a,e,f]
    
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
    
    a_accum = []
    b_accum = []
    c_accum = []
    d_accum = []
    e_accum = []
    f_accum = []
    g_accum = []
    no_iter = 20
    for j in range(no_iter):
        for k in f_node:
            k.support = sum([x.Glob_Thru for x in k.NonconnNodes])
            
        print('\n')
        print('a Thru = ',a.Thru)
        print('a Glob_Thru = ',a.Glob_Thru)
        print('b Thru = ',b.Thru)
        print('b Glob_Thru = ',b.Glob_Thru)
        print('c Thru = ',c.Thru)
        print('c Glob_Thru = ',c.Glob_Thru)
        print('d Thru = ',d.Thru)
        print('d Glob_Thru = ',d.Glob_Thru)
        print('e Thru = ',e.Thru)
        print('e Glob_Thru = ',e.Glob_Thru)
        print('f Thru = ',f.Thru)
        print('f Glob_Thru = ',f.Glob_Thru)
        print('g Thru = ',g.Thru)
        print('g Glob_Thru = ',g.Glob_Thru)
        print('\n')
        print('\n')
        
        
        temp_a = a.Thru
        temp_b = b.Thru
        temp_c = c.Thru
        temp_d = d.Thru
        temp_e = e.Thru
        temp_f = f.Thru
        temp_g = g.Thru
        a_accum.append(temp_a)
        b_accum.append(temp_b)
        c_accum.append(temp_c)
        d_accum.append(temp_d)
        e_accum.append(temp_e)
        f_accum.append(temp_f)
        g_accum.append(temp_g)

            
            
        temp_a = t_a if j>100 and j%10 == 1 else t_a*(gain + 0.5+0.5*np.tanh((0.1*j+1)*(a.Thru+a.Glob_Thru-max([x.Thru+x.Glob_Thru for x in a.connVNodes]) ))) 
        

        temp_b = t_b if j>100 and j%20 == 1 else t_b*(gain + 0.5+0.5*np.tanh((0.1*j+1)*(b.Thru+b.Glob_Thru-max([x.Thru+x.Glob_Thru for x in b.connVNodes]) ))) 
        
        
        temp_c = t_c if j>100 and j%30 == 1 else t_c*(gain + 0.5+0.5*np.tanh((0.1*j+1)*(c.Thru+c.Glob_Thru-max([x.Thru+x.Glob_Thru for x in c.connVNodes]) ))) 
        
        
        temp_d = t_d if j>100 and j%40 == 1 else t_d*(gain + 0.5+0.5*np.tanh((0.1*j+1)*(d.Thru+d.Glob_Thru-max([x.Thru+x.Glob_Thru for x in d.connVNodes]) ))) 
        
        temp_e = t_e if j>100 and j%50 == 1 else t_e*(gain + 0.5+0.5*np.tanh((0.1*j+1)*(e.Thru+e.Glob_Thru-max([x.Thru+x.Glob_Thru for x in e.connVNodes]) ))) 
        
        temp_f = t_f if j>100 and j%60 == 1 else t_f*(gain + 0.5+0.5*np.tanh((0.1*j+1)*(f.Thru+f.Glob_Thru-max([x.Thru+x.Glob_Thru for x in f.connVNodes]) ))) 
        
        temp_g = t_g if j>100 and j%70 == 1 else t_g*(gain + 0.5+0.5*np.tanh((0.1*j+1)*(g.Thru+g.Glob_Thru-max([x.Thru+x.Glob_Thru for x in g.connVNodes]) )))
        
        
        
        a.Thru = temp_a
        b.Thru = temp_b
        c.Thru = temp_c
        d.Thru = temp_d
        e.Thru = temp_e
        f.Thru = temp_f
        g.Thru = temp_g

        
        a.UpdateThru()
        b.UpdateThru()
        c.UpdateThru()
        d.UpdateThru()
        e.UpdateThru()
        f.UpdateThru()
        g.UpdateThru()
        
    print(a.Thru-gain*t_a)
    print(b.Thru-gain*t_b)
    print(c.Thru-gain*t_c)
    print(d.Thru-gain*t_d)
    print(e.Thru-gain*t_e)
    print(f.Thru-gain*t_f)
    print(g.Thru-gain*t_g)
    plt.figure(0)
    plt.plot([i for i in range(no_iter)], a_accum)
    plt.figure(1)
    plt.plot([i for i in range(no_iter)], b_accum)
    plt.figure(2)
    plt.plot([i for i in range(no_iter)], c_accum)
    plt.figure(3)
    plt.plot([i for i in range(no_iter)], d_accum)
    plt.figure(4)
    plt.plot([i for i in range(no_iter)], e_accum)
    plt.figure(5)
    plt.plot([i for i in range(no_iter)], f_accum)
    plt.figure(6)
    plt.plot([i for i in range(no_iter)], g_accum)
    plt.figure(7)
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