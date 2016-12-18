# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:36:50 2015

@author: hazem.soliman
"""

class Interv(object):
    
    def __init__(self,iden, a, b, weigh):
        self.iden = iden
        self.a = a
        self.b = b
        self.weigh = weigh
        self.psi = 0
        
        
class Resour(object):
    
    def __init__(self, iden):
        self.iden = iden
        self.starting = []
        self.ending = []
        
if __name__ == "__main__":
    
    List_Intervs = []    
    List_Intervs.append(Interv(1,2,8,6))
    List_Intervs.append(Interv(2,3,12,5))
    List_Intervs.append(Interv(3,1,14,8))
    List_Intervs.append(Interv(4,11,16,4))
    List_Intervs.append(Interv(5,15,17,7))
    List_Intervs.append(Interv(6,13,19,7))
    List_Intervs.append(Interv(7,18,22,6))
    List_Intervs.append(Interv(8,20,23,5))
    
    
    List_Res = []
    for i in range(24):
        r = Resour(i)
        for v in List_Intervs:
            if v.a == i:
                r.starting.append(v)
            if v.b == i:
                r.ending.append(v)
        List_Res.append(r)
       
    temp_max = 0
    last_interval = None
    List_of_sets = []
    for r in List_Res:
        for v in r.starting:
            v.psi = temp_max + v.weigh
        for v in r.ending:
            if v.psi > temp_max:
                temp_max = v.psi
                last_interval = v
        print("temp_max",temp_max)
        if last_interval:
            print("identity",last_interval.iden)
            
    Max_Indep_Set = []
    Max_Indep_Set.append(last_interval)
    temp_max_reverse = temp_max - last_interval.weigh
    for r in reversed(List_Res):
        if r.iden >= last_interval.a:
            continue
        for v in r.ending:
            if v.psi == temp_max_reverse:
                last_interval = v
                temp_max_reverse = temp_max_reverse - v.weigh
                Max_Indep_Set.append(v)
                
                
    for v in Max_Indep_Set:
        print(v.iden)
        
            