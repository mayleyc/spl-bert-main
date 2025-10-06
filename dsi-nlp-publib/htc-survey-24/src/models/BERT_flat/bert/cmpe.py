from pysdd.sdd import SddManager, Vtree

import json
from timeit import default_timer as timer
from itertools import combinations
import os

import torch.nn as nn

# Circuit imports
import sys
sys.path.append(os.path.join(sys.path[0],'src', 'hmc_utils'))
sys.path.append(os.path.join(sys.path[0],'src', 'hmc_utils', 'pypsdd'))

from src.hmc_utils.GatingFunction import DenseGatingFunction
from src.hmc_utils.compute_mpe import CircuitMPE
from src.models.BERT_flat.bert.bert_classifier import BERTForClassification, BERTForClassification_SPL


# misc
from common import *

def get_circuit(device, dataset_name, mat, size, num_st_nodes, S = 2, gates=2, num_reps=1):
    # Create constraints if not already done
    if not os.path.exists('constraints'):
        os.makedirs('constraints')
    
    # If constraints already exist, load them
    if not os.path.isfile('constraints/' + dataset_name + '_excl' + '.sdd') or not os.path.isfile('constraints/' + dataset_name + '_excl' + '.vtree'):
        # Compute matrix of ancestors R
        # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
        #np.savetxt("foo.csv", mat, delimiter=",") #Check mat
        R = np.zeros(mat.shape)
        np.fill_diagonal(R, 1)
        g = nx.DiGraph(mat)
        layer_map = layer_mapping_BFS(g.reverse(copy=True), num_st_nodes) # 1-indexed # keep original g
        #print(layer_map)
        #quit()
        for i in range(len(mat)):
            descendants = list(nx.descendants(g, i))
            if descendants:
                R[i, descendants] = 1
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
        nz_layers = {0} #bgc has Level 1 annotations :(

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
        #print_satisfying_configs(alpha, R.size(0))


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
        #print_satisfying_configs(alpha, R.size(0))


        alpha.save(str.encode('constraints/' + dataset_name + '_excl'+ '.sdd'))
        alpha.vtree().save(str.encode('constraints/' + dataset_name + '_excl'+ '.vtree'))
        
    # Create circuit object
    cmpe = CircuitMPE('constraints/' + dataset_name + '_excl'+ '.vtree', 'constraints/' + dataset_name + '_excl'+ '.sdd')
    

    if S > 0:
        cmpe.overparameterize(S=S)
        print("Done overparameterizing")

    # Create gating function
    gate = DenseGatingFunction(cmpe.beta, gate_layers=[size] + [256]*gates + [145], num_reps=num_reps).to(device)

    R = None
    return cmpe, gate, R

class SPLBERTModel(nn.Module):
    def __init__(self, cmpe, gate, base_model=BERTForClassification_SPL):
        super().__init__()
        self.device = next(base_model.parameters()).device
        self.base_model = base_model
        self.cmpe = cmpe
        self.gate = gate.to(self.device)
        
    def forward(self, data):
        x, labels = data
        '''print("x:", type(x)) #dict
        k, v = next(iter(x.items()))
        print("x item:", type(k), "\n", type(v)) #dict
        quit()'''
        l, y, *_ = self.base_model(data)
        #print("y:", type(y)) #tensor
        thetas = self.gate(l)
        #print("thetas:", type(thetas)) #list
        
        #thetas_cpu = [t.to("cpu") for t in thetas]  # list of tensors on device
        #y = y.to("cpu")

        #thetas = thetas.to(self.device)
        # INSERT CMPE HERE
        #print("thetas:", type(thetas)) #(bs, 145)
        self.cmpe.set_params(thetas)
        nll = self.cmpe.cross_entropy(y, log_space=True).mean()

        with torch.no_grad():
            pred = (self.cmpe.get_mpe_inst(x["input_ids"].shape[0])) #> 0).long()
        #print("pred:", type(pred)) #tensor
        #print("y:", type(y)) #tensor
        #pred = pred.to('cpu')
        return pred, y, nll, l
    
    def constructor_args(self):
        if hasattr(self.base_model, "constructor_args"):
            return self.base_model.constructor_args()
        else:
            return {}
