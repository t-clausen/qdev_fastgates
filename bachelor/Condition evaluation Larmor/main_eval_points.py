import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import sys
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.getcwd())
from utils.gate_params import get_gate_params

config = {}
with open("config.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        config[line.split(":")[0]] = eval(line.split(":")[1])

gate_names_2_eval = [#actual values are hidden off in the other file
    #"commensurate_x_virt_z_nooptim",#!dont use this one
    #"FAST-MAGNUS_nooptim",
    #"RWA_x_nooptim",
    #"FAST-DRAG",
    #"FAST-MAGNUS-DRAG",
    #"FAST-MAGNUS-MAGNUS-fix1707-amp-N3",
    #"FAST-MAGNUS-MAGNUS-fix1707-amp-woFAST-derrquad-N3",
    #"FAST-MAGNUS-MAGNUS-fix1707-amp-woFAST-N3",
    #"FAST-MAGNUS-MAGNUS-fix1707-amp-N2",
    #"FAST-MAGNUS-MAGNUS-fix1707-amp-N1",
    #"FAST-MAGNUS-MAGNUS-fix1707-amp-monoquad",
    #"FAST-MAGNUS-MAGNUS-fix1707-amp-dualquad",
    #"FAST-MAGNUS-MAGNUS-fix1707-amp-fullquad",
    #"FAST-MAGNUS-MAGNUS-fix1707-amp-derrquad",
    #"FAST-MAGNUS-MAGNUS-fix1707-hypFAST-derrquad-N3",
    #"FAST-MAGNUS-MAGNUS-fix1707-woFAST-amp-N7",
    #"FAST-MAGNUS-MAGNUS-fix1707-hypFAST-amp-N3",
    #"FAST-MAGNUS-MAGNUS-fix1707-hypFAST-N3",
    
    #"FAST-MAGNUS-MAGNUS-fix1707-woFAST-amp-N3",
    #"FAST-MAGNUS-MAGNUS-fix1707-woFAST-amp-N5",
    #"FAST-MAGNUS-MAGNUS-fix1707-woFAST-fix2307-derrquad-amp-N5",
    #"FAST-MAGNUS-MAGNUS-fix1707-woFAST-fix2307-amp-N5",
    "FAST-MAGNUS-MAGNUS-fix1707-hypFAST-fix2307-amp-N5",
    "FAST-MAGNUS-MAGNUS-fix1707-hypFAST-fix2307-derrquad-amp-N5",
    #"magnus1_x_virt_z_nooptim",
    
    
    #"FAST-MAGNUS-MAGNUS-fix1707-woFAST-amp-N10",
    #"FAST-MAGNUS-MAGNUS-fix1707-hypFAST-derrquad-N10",
    #"FAST-MAGNUS-MAGNUS-fix1707-hypFAST-N10",
    #"FAST-MAGNUS-MAGNUS-fix1707-hypFAST-amp-N10",
    #"FAST-MAGNUS-MAGNUS-fix1707-woFAST-derrquad-amp-N3",
    #"FAST-MAGNUS-MAGNUS-fix1707-woFAST-monoquad-amp-N3",
    #"FAST-MAGNUS-MAGNUS-fix1707-woFAST-dualquad-amp-N3",
    
    
    #"FAST-MAGNUS-MAGNUS-noFM",
    #"FAST-MAGNUS-MAGNUS-minus",
    #"FAST-MAGNUS-MAGNUS-amp",
    
    #"commensurate_x_virt_z",
    #"commensurate_x_virt_z_new",
    #"corotating_xy_virt_z_nooptim",
    

    
    
    
    #"magnus1_x_virt_z_nooptim",

    #"corotating_xy_virt_z",#!dont use this one
    
    #"FAST_nooptim",

    
    #"magnus1_x_virt_z",#!dont use this one

    
]

scores_withmeta = {}
if os.path.exists("scores_withmeta.pickle"):
    with open("scores_withmeta.pickle", "rb") as f:
        scores_withmeta = pickle.load(f)


from utils.eval_functions import *
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import Pool as normalPool
from copy import deepcopy as copy
from random import shuffle
from utils.eval_functions import AdaptiveLearner

#!todo: Have virtual-z include the full H_0 matrix
#!understand why RWA does not move under variation of alpha

def instantiate_learners(gate_name):
    learner = AdaptiveLearner(scores_withmeta[gate_name], [10, 1, 10, None, None, np.pi, (10,10000), 0, 0], framework="adaptive_area",truncation=2)
    return learner

"""alpha_ICs = {
    0.01: [0.01, 1, 0.5],
    0.1: [0.05, 1, 1.08],
    1: [0.27, 1, 2.4],
    10: [2, 1, 10],
    20: [4.02, 1, 21.7],
}
"""
def main():
    #instantiate adaptive learner
    learners = {}
    """for k in list(scores_withmeta.keys()):
        if "commensurate" in k and not "nooptim" in k:
            scores_withmeta["comm_old"] = scores_withmeta[k]
            del scores_withmeta[k]
    #pickle
    with open("scores_withmeta.pickle", "wb") as f:
        pickle.dump(scores_withmeta, f)"""
    
    while True:
        for t in range(2,6):
            alphas = np.logspace(np.log10(0.019), np.log10(20), 50)
            #alphas = np.logspace(np.log10(1), np.log10(20), 50)
            shuffle(alphas)
            #for alpha in np.logspace(np.log10(0.01), np.log10(20), 20):
            for alpha in alphas:
                #alpha = 20#!temp
                #alpha = 18
                #for alpha in [10]:
                #for alpha in [20]:
                
                for i in range(len(gate_names_2_eval)):
                    if gate_names_2_eval[i] not in scores_withmeta.keys():
                        scores_withmeta[gate_names_2_eval[i]] = []
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, (0.1, 10), (0.1, 10), np.pi, (0.1,10), 1.58e-4, 1.58e-4], framework="adaptive")
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, (0.58, 0.6), (5.70, 5.72), np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, 0.59, 5.71, np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, 0.59, (0.1,10), np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, 0.59, (0.1,10), np.pi, (10,30), 1.58e-4, 1.58e-4], framework="adaptive")
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, (0,2), 5.71, np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, np.linspace(0,10,5), 5.71, np.pi, (10,30), 0, 0], framework="adaptive",truncation=2)
                    
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [10, 1, 10, np.pi, (5,5.002), 0, 0], framework="adaptive",truncation=2)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [10, 1, 10, np.pi, (1,1.002), 0, 0], framework="adaptive",truncation=2)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1, 0.8, 10, np.pi, (0.001,100), 0, 0], framework="adaptive",truncation=10)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1, 1, 1, np.pi, (29.4,29.6), 0, 0], framework="adaptive",truncation=2)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 0.25, 25, np.pi, (40,5000), 0, 0], framework="adaptive_area",truncation=3)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 0.25, alpha, np.pi, (0.1,20), 0, 0], framework="adaptive_area",truncation=2)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 0.25, alpha, np.pi, (0.1,20), 0, 0], framework="random",truncation=t)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 0.25, alpha, np.pi, (1000,10000), 0, 0], framework="adaptive",truncation=2)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, alpha, np.pi, (0.2,80), 0, 0], framework="random",truncation=4)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, alpha, np.pi, (0.2,80), 30e3, 20e3], framework="random",truncation=4)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, alpha, np.pi, (0.2,20), 30e3, 20e3], framework="random",truncation=5)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, alpha, np.pi, (20,80), 30e3, 20e3], framework="random",truncation=5)
                    #alpha = 20
                    scores_tmp = scores_withmeta[gate_names_2_eval[i]].copy()
                    for j in range(len(scores_tmp)):
                        scores_tmp[j]["alpha_target"] = np.log10(scores_tmp[j]["alpha_target"])
                        scores_tmp[j]["Lambdas"] = np.log10(scores_tmp[j]["Lambdas"])
                    learner = AdaptiveLearner(scores_tmp, [None, 1, None, 1, (np.log10(0.019),np.log10(20)), np.pi, (np.log10(0.2),np.log10(80)), np.inf, np.inf], framework="random",truncation=5)
                    #learner = AdaptiveLearner(scores_tmp, [None, 1, None, 1, np.log10(alpha), np.pi, (np.log10(0.2),np.log10(80)), np.inf, np.inf], framework="random",truncation=5)
                    print(alpha)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, (np.log10(0.019),np.log10(20)), np.pi, (np.log10(0.2),np.log10(20)), np.inf, np.inf], framework="random",truncation=5)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, alpha, np.pi, (0.2,80), np.inf, np.inf], framework="random",truncation=5)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, alpha, np.pi, (0.99,1.01), np.inf, np.inf], framework="random",truncation=5)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, alpha, np.pi, (0.2,20), np.inf, np.inf], framework="random",truncation=2)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, alpha, np.pi, (0.2,10), 30e3, 20e3], framework="random",truncation=2)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 1, alpha, np.pi, (0.2,80), np.inf, np.inf], framework="random",truncation=2)
                    #learner = AdaptiveLearner([], [None, 1, None, 1, alpha, np.pi, (0.2,80), 30e3, 20e3], framework="random",truncation=4)
                    #learner = AdaptiveLearner([], [None, 1, None, 1, alpha, np.pi, (79,80), 30e3, 20e3], framework="random",truncation=4)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 0.25, alpha, np.pi, (1,2), 0, 0], framework="adaptive_area",truncation=3)
                    #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 0.25, 25, np.pi, (10,10.01), 0, 0], framework="adaptive_area",truncation=3)
                    learners[gate_names_2_eval[i]] = learner
                #pool the above task
                
                """with normalPool(processes=1) as pool:#len(gate_names_2_eval)) as pool:
                    results = pool.map(instantiate_learners, gate_names_2_eval)
                    for i in range(len(gate_names_2_eval)):
                        learners[gate_names_2_eval[i]] = results[i]"""
                pointscnt = [len(scores_withmeta[gate_names_2_eval[i]]) if gate_names_2_eval[i] in scores_withmeta.keys() else 0 for i in range(len(gate_names_2_eval))]
                lowest = np.argsort(pointscnt)[:2]
                for k,gate_name in enumerate(gate_names_2_eval):
                    if k not in lowest:
                        continue
                    #get the next point to evaluate
                    learner = learners[gate_name]
                    qubits_2_eval = learner.get_next_dp(N=60)
                    for i in range(len(qubits_2_eval)):
                        qubits_2_eval[i]["alpha_target"] = np.power(10, qubits_2_eval[i]["alpha_target"])
                        qubits_2_eval[i]["Lambdas"] = np.power(10, qubits_2_eval[i]["Lambdas"])
                    if "comm" in gate_name:# in this case, the gate should only be performed on integers of a certain periodicity.
                        period = 0.5
                        for i in range(len(qubits_2_eval)):
                            qubits_2_eval[i]["Lambdas"] = np.ceil(qubits_2_eval[i]["Lambdas"] / period) * period
                        #take only unique values of Lambdas
                        lambdas = [q["Lambdas"] for q in qubits_2_eval]
                        qubits_2_eval = [qubits_2_eval[i] for i in range(len(qubits_2_eval)) if qubits_2_eval[i]["Lambdas"] not in lambdas[:i]]

                    shuffle(qubits_2_eval)
                    """for i in range(len(qubits_2_eval)):
                        a = qubits_2_eval[i]["alpha_target"]
                        idx_IC = np.argmin(np.abs(np.array(list(alpha_ICs.keys())) - a))
                        Ec_IC = alpha_ICs[list(alpha_ICs.keys())[idx_IC]][0]
                        El_IC = alpha_ICs[list(alpha_ICs.keys())[idx_IC]][1]
                        Ej_IC = alpha_ICs[list(alpha_ICs.keys())[idx_IC]][2]
                        qubits_2_eval[i]["Ec_IC"] = Ec_IC
                        qubits_2_eval[i]["El_IC"] = El_IC
                        qubits_2_eval[i]["Ej_IC"] = Ej_IC"""

                    gate_instances = []
                    gate_params = get_gate_params(gate_name)
                    for i in range(len(qubits_2_eval)):
                        gate_instances.append(copy(gate_params))
                    print(f"Evaluating {gate_name} with {len(qubits_2_eval)} qubits")
                    #_, _, _, _, _, _, _, _, _, _ = qbi.init_qubit(qubits_2_eval[0]["Ec"], qubits_2_eval[0]["El"], qubits_2_eval[0]["Ej"], qubits_2_eval[0]["phi_dc"],qubits_2_eval[0]['omega_01_target'],qubits_2_eval[0]['alpha_target'], qubits_2_eval[0]["c_ops"], qubits_2_eval[0]["Lambdas"],truncation=qubits_2_eval[0]["truncation"], base_ex=qubits_2_eval[0]["base_ex"], base_size=qubits_2_eval[0]["base_size"],Ec_IC=qubits_2_eval[0]["Ec_IC"],El_IC=qubits_2_eval[0]["El_IC"],Ej_IC=qubits_2_eval[0]["Ej_IC"])
                    if 0:
                        for i,qubit,gate in zip(range(len(qubits_2_eval)), qubits_2_eval, gate_instances):
                            qubits_2_eval[i], gate_instances[i] = calib_gate((qubit, gate))
                    else:
                        results = []
                        with normalPool(processes=12) as pool:
                            args = [(qubits_2_eval[i], gate_instances[i]) for i in range(len(qubits_2_eval))]
                            results = pool.map(calib_gate, args)
                            for i in range(len(qubits_2_eval)):
                                #gate_instances[i] = results[i]
                                qubits_2_eval[i], gate_instances[i] = results[i]
                    print("Starting evaluation")
                    if 0:
                        results = [do_test(q,g) for q,g in zip(qubits_2_eval, gate_instances)]
                    else:
                        results = []
                        with ProcessPool(max_workers=14) as pool:
                            future = pool.map(do_test, qubits_2_eval, gate_instances, timeout=60*60)#!conservative
                            iter = future.result()
                            for i in range(len(qubits_2_eval)): 
                                def get_null_result(q,timeout=False):
                                    result = [[None], None]
                                    result[1] = [q]
                                    result[1][0]["score"] = None
                                    if timeout:
                                        result[1][0]["timeout"] = True
                                    return result
                                try:
                                    result = next(iter)
                                    results.append(result)
                                except ValueError as e:
                                    print(f"ValueError: {e}")
                                    results.append(get_null_result(qubits_2_eval[i]))
                                except TimeoutError:
                                    print("Timeout")
                                    results.append(get_null_result(qubits_2_eval[i], timeout=True))
                                except:
                                    print("Unknown error")
                    #unpack the results
                    for i,result in enumerate(results):
                        scores, points = result
                        if gate_name not in scores_withmeta.keys():
                            scores_withmeta[gate_name] = []
                        for j,point in enumerate(points):
                            #point["gate"] = gate_name
                            scores_withmeta[gate_name].append(point)
                    with open(f"scores_withmeta.pickle", "wb") as f:
                        pickle.dump(scores_withmeta, f)
                    #feed the datapoints to the learner
                    learner.feed_points(scores_withmeta[gate_name])
                    learners[gate_name] = learner
                    


if __name__ == "__main__":
    main()