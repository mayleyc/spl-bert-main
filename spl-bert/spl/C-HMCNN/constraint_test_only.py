import os
import datetime
import json
from time import perf_counter
import copy
import pickle
import glob
from itertools import combinations


import torch
import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter


from sklearn.metrics import (
    precision_score, 
    average_precision_score, 
    hamming_loss, 
    jaccard_score
)
from sklearn.model_selection import train_test_split

import json
from timeit import default_timer as timer


# Circuit imports
import sys
sys.path.append(os.path.join(sys.path[0],'hmc-utils'))
sys.path.append(os.path.join(sys.path[0],'hmc-utils', 'pypsdd'))

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE
from pysdd.sdd import SddManager, Vtree

from sklearn import preprocessing


# misc
from common import *


def log1mexp(x):
        assert(torch.all(x >= 0))
        return torch.where(x < 0.6931471805599453094, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

from torch.utils.data import Dataset
from PIL import Image

def print_satisfying_configs(alpha, num_vars):
    """Print all satisfying assignments as lists of active node indices"""
    print("Current satisfying configs:")
    for model in alpha.models():  # iterates over satisfying assignments
        active_nodes = [i for i in range(num_vars) if model.get(i+1, False)]
        print(active_nodes)
    print("---")


mat = np.load(mat_path_dict["custom"])




# Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
#np.savetxt("foo.csv", mat, delimiter=",") #Check mat
R = np.zeros(mat.shape)
np.fill_diagonal(R, 1)
g = nx.DiGraph(mat)
layer_map = layer_mapping_BFS(g.reverse(copy=True), num_st_nodes) # 1-indexed # keep original g
#print(layer_map)
#quit()
for i in range(len(mat)):
    #descendants = list(nx.descendants(g, i))
    ancestors = list(nx.ancestors(g, i))

    #if descendants:
        #R[i, descendants] = 1
    if ancestors:
        R[i, ancestors] = 1 #since j is ancestor of i
R = torch.tensor(R)

#Transpose to get the ancestors for each node 
R = R.unsqueeze(0).to(device)

# Uncomment below to compile the constraint
R.squeeze_()
mgr = SddManager(
    var_count=R.size(0),
    auto_gc_and_minimize=True)

max_layer = max(layer_map.values())

me_layers = layer_map.values() #{l for l in layer_map.values() if l != max_layer} #{l for l in layer_map.values() if l != max_layer} #{max_layer-1, max_layer} #layer_map.values() #
nz_layers = layer_map.values() #{0} #bgc has Level 1 annotations :(
# there is no root in R

'''alpha = mgr.true()
alpha.ref()
for i in range(R.size(0)):

    beta = mgr.true()
    beta.ref()
    for j in range(R.size(0)):

        if R[i][j] and i != j:
            old_beta = beta
            beta = beta & mgr.vars[j+1]
            beta.ref()
            old_beta.deref()

    old_beta = beta
    beta = -mgr.vars[i+1] | beta
    beta.ref()
    old_beta.deref()

    #DELTA - ME constraint
    if layer_map[i] not in me_layers:
        continue
    #make a list of indices of all species under g
    species = [j for j in range(R.size(0)) if R[i][j] and i != j]

    for idx1 in range(len(species)):
        for idx2 in range(idx1 + 1, len(species)): # all species after s1

            old_beta = beta
            beta = beta & (-mgr.vars[idx1+1] | -mgr.vars[idx2+1]) #one clause must be true # sdd count starts at 1
            beta.ref()
            old_beta.deref()


    old_alpha = alpha
    alpha = alpha & beta
    alpha.ref()
    old_alpha.deref()
'''

alpha = mgr.true()
alpha.ref()

print("alpha before anything:", alpha.is_true(), alpha.is_false(), alpha.model_count())

for layer in set(nz_layers) | set(me_layers):
    layer_nodes = [k for k in range(len(layer_map)) if layer_map[k] == layer]

    if not layer_nodes:
        continue

    zeta = mgr.true(); zeta.ref()   # default true if not needed
    delta = mgr.true(); delta.ref()

    if layer in nz_layers:
        old_zeta = zeta
        zeta = mgr.false(); zeta.ref() # initialize for NZ disjunction
        old_zeta.deref()
        for k in layer_nodes:
            old_zeta = zeta
            zeta = zeta | mgr.vars[k+1] # cumulative
            zeta.ref()
            old_zeta.deref()

    if layer in me_layers:
        old_delta = delta
        delta = mgr.true(); delta.ref()
        old_delta.deref()
        for s1, s2 in combinations(layer_nodes, 2):
            old_delta = delta
            delta = delta & (-mgr.vars[s1+1] | -mgr.vars[s2+1]) # only 1 per pair
            delta.ref()
            old_delta.deref()

    me_nz = delta & zeta
    me_nz.ref()

    old_alpha = alpha
    alpha = alpha & me_nz
    alpha.ref()
    old_alpha.deref()

print("alpha after both delta & zeta hierarchy:", alpha.is_true(), alpha.is_false(), alpha.model_count())
print_satisfying_configs(alpha, R.size(0))


for i in range(R.size(0)): # ONE CHILD PER LEVEL
    beta = mgr.false()
    beta.ref()

    has_child = False
    for j in range(R.size(1)):

        if R[i][j] and i != j: # why not R[j][i]?
            has_child = True
            old_beta = beta
            beta = beta | mgr.vars[j+1] # conjunction of all of its children: true & j1 & j2 & ...
            beta.ref()
            old_beta.deref()

    if has_child:
        old_beta = beta
        beta = -mgr.vars[i+1] | beta
        beta.ref()
        old_beta.deref()
    else:
        # leaf: parent allowed without restriction
        old_beta = beta
        beta = mgr.true()
        beta.ref()
        old_beta.deref()

    old_alpha = alpha
    alpha = alpha & beta
    alpha.ref()
    old_alpha.deref()

print("alpha after beta hierarchy:", alpha.is_true(), alpha.is_false(), alpha.model_count()) #25 because no ME
print_satisfying_configs(alpha, R.size(0))

'''# NONZERO CONSTRAINTS HERE
for layer in nz_layers:
    layer_nodes = [k for k in range(len(layer_map)) if layer_map[k] == layer]
    if layer_nodes:
        # enforce at least one True in the layer
        zeta = mgr.false()
        zeta.ref()
        for k in layer_nodes:
            old_zeta = zeta
            zeta = zeta | mgr.vars[k+1]
            zeta.ref()
            old_zeta.deref()


# --- MUTUAL EXCLUSION LAYERS (at most one node) ---
for layer in me_layers:
    layer_nodes = [k for k in range(len(layer_map)) if layer_map[k] == layer]
    if layer_nodes:

        delta = mgr.true()
        delta.ref()

        for s1, s2 in combinations(layer_nodes, 2):
            old_delta = delta
            delta = delta & (-mgr.vars[s1+1] | -mgr.vars[s2+1])
            delta.ref()
            old_delta.deref()

        me_nz = delta & zeta
        me_nz.ref()      

        old_alpha = alpha
        alpha = alpha & me_nz
        alpha.ref()
        old_alpha.deref()'''

'''# enforce mutual exclusivity: at most one can be True
    for s1, s2 in combinations(layer_nodes, 2):
        old_alpha = alpha
        alpha = alpha & (mgr.vars[s1+1]| mgr.vars[s2+1])
        alpha.ref()
        old_alpha.deref()'''

'''for i in range(R.size(0)):
    # initialize for every node
    zeta = mgr.false() 
    zeta.ref()
    layer_nodes = [k for k in range(len(layer_map)) if layer_map[k] == layer_map[i]]
    
    #(R[i] == 1).nonzero(as_tuple=True)[0].tolist() #
    if layer_map[i] in nz_layers:
        
        species_nz = [j for j in range(R.size(1)) if R[j][i] and layer_map[j] == layer_map[i] + 1]
        #species_nz = [j for j in range(R.size(0)) if R[i][j]] #and layer_map[j] == layer_map[i] + 1] #PROBLEM HERE: THIS IS [0] 
        #print(species_nz) #[0]
        #quit()

        if species_nz:
        # RESULTS: alpha after zeta hierarchy: 0 0 479001600
        #if species_nz:
        #old_zeta = zeta
            zeta = mgr.vars[species_nz[0]+1]
            zeta.ref()
            #old_zeta.deref()
            for s in species_nz[1:]: # loop to OR every species
                old_zeta = zeta
                # A -> B => -A | A = -A | B
                #check for 1 single parent-child pair (from beta downwards): should go down by 1
                #Write down on paper the truth table
                #without constraints: 2^n + 1
                
                zeta = zeta | mgr.vars[s+1]                            
                zeta.ref()
                old_zeta.deref()

        for other_i in layer_nodes: #enforce ME on the parent too
            if other_i != i:
                old_zeta = zeta
                zeta = zeta | mgr.vars[other_i+1]
                zeta.ref()
                old_zeta.deref()

        old_zeta = zeta
        zeta = -mgr.vars[i+1] | zeta
        zeta.ref()
        old_zeta.deref()

        old_alpha = alpha
        alpha = alpha & zeta
        alpha.ref()
        old_alpha.deref() 

    zeta.deref()'''




# ME constraint

'''# within each layer first
for layer in me_layers:
    layer_nodes = [k for k in range(len(layer_map)) if layer_map[k] == layer]
# enforce mutual exclusivity: at most one can be True
    for s1, s2 in combinations(layer_nodes, 2):
        old_alpha = alpha
        alpha = alpha & (-mgr.vars[s1+1] | -mgr.vars[s2+1])
        alpha.ref()
        old_alpha.deref()'''
        
'''    
# then initialize delta clause 
for i in range(R.size(0)):    
    if layer_map[i] in me_layers:
        #make a list of indices of all species under g
        
        children = [j for j in range(R.size(1)) if R[j][i] and layer_map[j] == layer_map[i] + 1] #i != j]
        if children:
            delta = mgr.false() 
            delta.ref()
            for child in children:
                old_delta = delta
                delta = delta | mgr.vars[child+1]
                delta.ref()
                old_delta.deref()

            old_alpha = alpha
            alpha = alpha & (-mgr.vars[i+1] | delta)
            alpha.ref()
            old_alpha.deref()
            delta.deref()
            
            for s1, s2 in combinations(children, 2): # all species after s1  

                old_delta = delta
                delta = delta & (-mgr.vars[s1+1] | -mgr.vars[s2+1]) #OR: one clause must be true # sdd count starts at 1                       
                #delta = delta & (-mgr.vars[idx1+1] | -mgr.vars[idx2+1]) #OR: one clause must be true # sdd count starts at 1
                delta.ref()
                old_delta.deref()
            
            for child in children:
                old_alpha = alpha
                alpha = alpha & (-mgr.vars[child+1] | mgr.vars[i+1])
                alpha.ref()
                old_alpha.deref()

        
            old_delta = delta
            delta = -mgr.vars[i+1] | delta
            delta.ref()
            old_delta.deref()

            old_alpha = alpha
            alpha = alpha & delta
            alpha.ref()
            old_alpha.deref()'''
    

# print("alpha after delta hierarchy:", alpha.is_true(), alpha.is_false(), alpha.model_count())
# print_satisfying_configs(alpha, R.size(0))




#print("zeta (nonzero):", zeta.is_true(), zeta.is_false(), zeta.model_count())      

    # Mutual exclusivity logic
#try with 1 parent: see if the numbers are correct


'''for i in range(R.size(0)): #for all genera g
    #initialize delta for ME
    delta = mgr.true()
    delta.ref()
    if layer_map[i] not in me_layers:
        continue
    #make a list of indices of all species under g
    species = [j for j in range(R.size(0)) if R[i][j] and layer_map[j] == layer_map[i] + 1] #i != j]
    if not species:
        continue

    for idx1 in range(len(species)):
        for idx2 in range(idx1 + 1, len(species)): # all species after s1
            s1 = species[idx1]   # actual node index
            s2 = species[idx2]   

            old_delta = delta
            delta = delta & (-mgr.vars[s1+1] | -mgr.vars[s2+1]) #OR: one clause must be true # sdd count starts at 1                       
            #delta = delta & (-mgr.vars[idx1+1] | -mgr.vars[idx2+1]) #OR: one clause must be true # sdd count starts at 1
            delta.ref()
            old_delta.deref()

    old_alpha = alpha
    alpha = alpha & delta
    alpha.ref()
    old_alpha.deref()
    

print("alpha after delta hierarchy:", alpha.is_true(), alpha.is_false(), alpha.model_count())

# NONZERO CONSTRAINTS HERE


for i in range(R.size(0)): 
    zeta = mgr.false()
    zeta.ref()
    if layer_map[i] not in nz_layers:
        continue
    
    species_nz = [j for j in range(R.size(0)) if R[i][j] and layer_map[j] == layer_map[i] + 1]
    if not species_nz:
        continue
    for s in species_nz: # loop to OR every species
        old_zeta = zeta
        # A -> B => -A | A = -A | B
        #check for 1 single parent-child pair (from beta downwards): should go down by 1
        #Write down on paper the truth table
        #without constraints: 2^n + 1
        zeta = zeta | mgr.vars[s+1] #select one #negative?
        zeta.ref()
        old_zeta.deref()

    old_alpha = alpha
    alpha = alpha & zeta
    alpha.ref()
    old_alpha.deref()   '''            






#print("delta (mutual exclusivity):", delta.is_true(), delta.is_false(), delta.model_count())

#print("zeta (nonzero):", zeta.is_true(), zeta.is_false(), zeta.model_count())