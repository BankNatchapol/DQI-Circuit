import sys
sys.path.append("..")
from decoders import *

# 8-bit code
G = np.array([[1, 0, 0, 0, 1, 0, 0, 1],
              [0, 1, 0, 0, 1, 1, 0, 0],
              [0, 0, 1, 0, 0, 1, 1, 0],
              [0, 0, 0, 1, 0, 0, 1, 1]])
H = np.array([[1, 1, 0, 0, 1, 0, 0, 0],
              [0, 1, 1, 0, 0, 1, 0, 0],
              [0, 0, 1, 1, 0, 0, 1, 0],
              [1, 0, 0, 1, 0, 0, 0, 1]])

code = LinearCode(G, H)



# parameters for data generation
N = 40
ch_angles = np.linspace(0., 0.5*np.pi, N)[1:]

data_h1 = list()
data_h2 = list()
data_h3 = list()
data_ideal_classical = list()
data_ideal_quantum = list()
data_strat1 = list()
data_strat2 = list()
data_strat3 = list()


for theta in ch_angles:
    print(theta)
    # automatically unrolled BPQM
    #############################################
    data_h1.append( decode_bpqm(code, theta, VarNodeCloner(theta), 1, 'bit', bit=0, only_zero_codeword=True, debug=False) )
    data_h2.append( decode_bpqm(code, theta, VarNodeCloner(theta), 2, 'bit', bit=0, only_zero_codeword=True, debug=False) )
    data_h3.append( decode_bpqm(code, theta, VarNodeCloner(theta), 3, 'bit', bit=0, only_zero_codeword=True, debug=False) )
    data_ideal_classical.append( decode_bit_optimal_classical(code, theta, 0) )
    data_ideal_quantum.append( decode_bit_optimal_quantum(code,theta, 0) )


    # Strategy 1: cut off x2
    #############################################
    # generate computational graph
    graph = nx.DiGraph()
    occurances = dict()
    for i in [0,1,3,4,5,6,7]: # variables
        graph.add_node("x"+str(i)+"_0", type="variable")
        graph.add_node("y"+str(i)+"_0", type="output")
        graph.add_edge("x"+str(i)+"_0", "y"+str(i)+"_0")
        occurances["x"+str(i)] = 1
    occurances["x2"] = 0
    for i in [0,1,2,3]: graph.add_node("c"+str(i), type="check")
    graph.add_edge("x0_0", "c0")
    graph.add_edge("x0_0", "c3")
    graph.add_edge("c0", "x4_0")
    graph.add_edge("c0", "x1_0")
    graph.add_edge("c3", "x7_0")
    graph.add_edge("c3", "x3_0")

    graph.add_edge("x1_0", "c1")
    graph.add_edge("c1", "x5_0")
    graph.add_edge("x3_0", "c2")
    graph.add_edge("c2", "x6_0")
    # qubit mapping + angle 
    qubit_mapping = {
        "y0_0" : 0, "y1_0" : 1, "y2_0" : 2, "y3_0" : 3, "y4_0" : 4, "y5_0" : 5, "y6_0" : 6, "y7_0" : 7
    }
    cloner = VarNodeCloner(theta)
    cloner.mark_angles(graph, occurances)
    for l in [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]: graph.nodes[l]["qubit_idx"] = qubit_mapping[l]
    # generate quantum circuit
    qc_cloner = cloner.generate_cloner_circuit(graph, occurances, qubit_mapping, 8)
    prob = 0.
    for cw in code.get_codewords(): 
        qc = QuantumCircuit(8)
        for i in range(8): qc.initialize(np.array([np.cos(0.5*theta),  (-1)**cw[i] * np.sin(0.5*theta)]), [i])
        qc += qc_cloner
        idx, angles = tree_bpqm(graph, qc, root="x0_0")
        qc.h(idx)
        qc.snapshot_probabilities('prob', qubits=[idx])
        # run simulation
        backend = QasmSimulator(method='statevector')
        job = execute(qc, backend=backend)
        vals = job.result().data()['snapshots']['probabilities']['prob'][0]['value']
        if '0x0' not in vals: vals['0x0'] = 0.
        if '0x1' not in vals: vals['0x1'] = 0.
        if int(cw[0]) == 0: prob += vals['0x0'] / 2**code.k
        else: prob += vals['0x1'] / 2**code.k
    data_strat1.append(prob)

    # Strategy 2: cut off x2 on one side
    #############################################
    # generate computational graph
    graph = nx.DiGraph()
    occurances = dict()
    for i in [0,1,2,3,4,5,6,7]: # variables
        graph.add_node("x"+str(i)+"_0", type="variable")
        graph.add_node("y"+str(i)+"_0", type="output")
        graph.add_edge("x"+str(i)+"_0", "y"+str(i)+"_0")
        occurances["x"+str(i)] = 1
    for i in [0,1,2,3]: graph.add_node("c"+str(i), type="check")
    graph.add_edge("x0_0", "c0")
    graph.add_edge("x0_0", "c3")
    graph.add_edge("c0", "x4_0")
    graph.add_edge("c0", "x1_0")
    graph.add_edge("c3", "x7_0")
    graph.add_edge("c3", "x3_0")

    graph.add_edge("x1_0", "c1")
    graph.add_edge("c1", "x5_0")
    graph.add_edge("c1", "x2_0")
    graph.add_edge("x3_0", "c2")
    graph.add_edge("c2", "x6_0")
    # qubit mapping + angle 
    qubit_mapping = {
        "y0_0" : 0, "y1_0" : 1, "y2_0" : 2, "y3_0" : 3, "y4_0" : 4, "y5_0" : 5, "y6_0" : 6, "y7_0" : 7
    }
    cloner = VarNodeCloner(theta)
    cloner.mark_angles(graph, occurances)
    for l in [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]: graph.nodes[l]["qubit_idx"] = qubit_mapping[l]
    # generate quantum circuit
    qc_cloner = cloner.generate_cloner_circuit(graph, occurances, qubit_mapping, 8)
    prob = 0.
    for cw in code.get_codewords(): 
        qc = QuantumCircuit(8)
        for i in range(8): qc.initialize(np.array([np.cos(0.5*theta),  (-1)**cw[i] * np.sin(0.5*theta)]), [i])
        qc += qc_cloner
        idx, angles = tree_bpqm(graph, qc, root="x0_0")
        qc.h(idx)
        qc.snapshot_probabilities('prob', qubits=[idx])
        # run simulation
        backend = QasmSimulator(method='statevector')
        job = execute(qc, backend=backend)
        vals = job.result().data()['snapshots']['probabilities']['prob'][0]['value']
        if '0x0' not in vals: vals['0x0'] = 0.
        if '0x1' not in vals: vals['0x1'] = 0.
        if int(cw[0]) == 0: prob += vals['0x0'] / 2**code.k
        else: prob += vals['0x1'] / 2**code.k
    data_strat2.append(prob)



    # Strategy 3: cut off check+x5
    #############################################
    # generate computational graph
    graph = nx.DiGraph()
    occurances = dict()
    for i in [0,1,2,3,4,5,6,7]: # variables
        graph.add_node("x"+str(i)+"_0", type="variable")
        graph.add_node("y"+str(i)+"_0", type="output")
        graph.add_edge("x"+str(i)+"_0", "y"+str(i)+"_0")
        occurances["x"+str(i)] = 1
    for i in [0,2,3]: graph.add_node("c"+str(i), type="check")
    graph.add_edge("x0_0", "c0")
    graph.add_edge("x0_0", "c3")
    graph.add_edge("c0", "x4_0")
    graph.add_edge("c0", "x1_0")
    graph.add_edge("c3", "x7_0")
    graph.add_edge("c3", "x3_0")

    graph.add_edge("x3_0", "c2")
    graph.add_edge("c2", "x6_0")
    graph.add_edge("c2", "x2_0")
    # qubit mapping + angle 
    qubit_mapping = {
        "y0_0" : 0, "y1_0" : 1, "y2_0" : 2, "y3_0" : 3, "y4_0" : 4, "y5_0" : 5, "y6_0" : 6, "y7_0" : 7
    }
    cloner = VarNodeCloner(theta)
    cloner.mark_angles(graph, occurances)
    for l in [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]: graph.nodes[l]["qubit_idx"] = qubit_mapping[l]
    # generate quantum circuit
    qc_cloner = cloner.generate_cloner_circuit(graph, occurances, qubit_mapping, 8)
    prob = 0.
    for cw in code.get_codewords(): 
        qc = QuantumCircuit(8)
        for i in range(8): qc.initialize(np.array([np.cos(0.5*theta),  (-1)**cw[i] * np.sin(0.5*theta)]), [i])
        qc += qc_cloner
        idx, angles = tree_bpqm(graph, qc, root="x0_0")
        qc.h(idx)
        qc.snapshot_probabilities('prob', qubits=[idx])
        # run simulation
        backend = QasmSimulator(method='statevector')
        job = execute(qc, backend=backend)
        vals = job.result().data()['snapshots']['probabilities']['prob'][0]['value']
        if '0x0' not in vals: vals['0x0'] = 0.
        if '0x1' not in vals: vals['0x1'] = 0.
        if int(cw[0]) == 0: prob += vals['0x0'] / 2**code.k
        else: prob += vals['0x1'] / 2**code.k
    data_strat3.append(prob)


    # reference h=1 
    #############################################
    # generate computational graph
    graph = nx.DiGraph()
    occurances = dict()
    for i in [0,1,3,4,7]: # variables
        graph.add_node("x"+str(i)+"_0", type="variable")
        graph.add_node("y"+str(i)+"_0", type="output")
        graph.add_edge("x"+str(i)+"_0", "y"+str(i)+"_0")
        occurances["x"+str(i)] = 1
    occurances["x2"] = 0
    occurances["x5"] = 0
    occurances["x6"] = 0
    for i in [0,2,3]: graph.add_node("c"+str(i), type="check")
    graph.add_edge("x0_0", "c0")
    graph.add_edge("x0_0", "c3")
    graph.add_edge("c0", "x4_0")
    graph.add_edge("c0", "x1_0")
    graph.add_edge("c3", "x7_0")
    graph.add_edge("c3", "x3_0")

    # qubit mapping + angle 
    qubit_mapping = {
        "y0_0" : 0, "y1_0" : 1, "y2_0" : 2, "y3_0" : 3, "y4_0" : 4, "y5_0" : 5, "y6_0" : 6, "y7_0" : 7
    }
    cloner = VarNodeCloner(theta)
    cloner.mark_angles(graph, occurances)
    for l in [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]: graph.nodes[l]["qubit_idx"] = qubit_mapping[l]
    # generate quantum circuit
    qc_cloner = cloner.generate_cloner_circuit(graph, occurances, qubit_mapping, 8)
    prob = 0.
    for cw in code.get_codewords(): 
        qc = QuantumCircuit(8)
        for i in range(8): qc.initialize(np.array([np.cos(0.5*theta),  (-1)**cw[i] * np.sin(0.5*theta)]), [i])
        qc += qc_cloner
        idx, angles = tree_bpqm(graph, qc, root="x0_0")
        qc.h(idx)
        qc.snapshot_probabilities('prob', qubits=[idx])
        # run simulation
        backend = QasmSimulator(method='statevector')
        job = execute(qc, backend=backend)
        vals = job.result().data()['snapshots']['probabilities']['prob'][0]['value']
        if int(cw[0]) == 0: prob += vals['0x0'] / 2**code.k
        else: prob += vals['0x1'] / 2**code.k
    #print(prob)


np.savez('data.npz', ch_angles, data_h1, data_h2, data_h3, data_ideal_classical, data_ideal_quantum, data_strat1, data_strat2, data_strat3)

