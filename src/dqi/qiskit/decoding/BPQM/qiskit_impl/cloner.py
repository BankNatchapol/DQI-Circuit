"""Cloning utilities used by the BPQM decoders."""

from typing import Dict, List, Tuple

import numpy as np
import networkx as nx
import qiskit

# Use the actual Qiskit QuantumCircuit
QuantumCircuit = qiskit.QuantumCircuit


class Cloner:
    """Base class for all cloner implementations."""

    def mark_angles(self, graph: nx.Graph, occurances: Dict[str, int]) -> None:
        """Annotate ``graph`` with angles for cloning."""
        raise NotImplementedError

    def generate_cloner_circuit(
        self, graph: nx.Graph, occurances: Dict[str, int], qubit_mapping: Dict[str, int], n_qubits: int,
    ) -> QuantumCircuit:
        """Return the circuit implementing the required cloning operations."""
        raise NotImplementedError

class VarNodeCloner(Cloner):
    """Simple cloner treating all outputs symmetrically."""

    def __init__(self, theta: float) -> None:
        self.theta = theta

    def mark_angles(self, graph: nx.Graph, occurances: Dict[str, int]) -> None:
        leaves = [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]
        for l in leaves:
            n = occurances[l.split("_")[0].replace("y","x")]
            t = np.arccos(np.cos(self.theta)**(1./n))
            graph.nodes[l]["angle"] = [(t,{})]

    def cloner_unitary(self, t1: float, t2: float) -> np.ndarray:
        aplus = (1./np.sqrt(2.)) * (np.cos(0.5*(t1-t2)) + np.cos(0.5*(t1+t2))) \
                     / np.sqrt(1. + np.cos(t1)*np.cos(t2))
        amin  = (1./np.sqrt(2.)) * (np.cos(0.5*(t1-t2)) - np.cos(0.5*(t1+t2))) \
                     / np.sqrt(1. + np.cos(t1)*np.cos(t2))
        bplus = (1./np.sqrt(2.)) * (np.sin(0.5*(t1+t2)) - np.sin(0.5*(t1-t2))) \
                     / np.sqrt(1. - np.cos(t1)*np.cos(t2))
        bmin  = (1./np.sqrt(2.)) * (np.sin(0.5*(t1+t2)) + np.sin(0.5*(t1-t2))) \
                     / np.sqrt(1. - np.cos(t1)*np.cos(t2))
        return np.array([
            [aplus, 0.,    0.,     amin   ],
            [amin,  0.,    0.,     -aplus ],
            [0.,    bplus, bmin,   0.     ],
            [0.,    bmin,  -bplus, 0.     ]
        ]).T

    def generate_cloner_circuit(
        self,
        graph: nx.Graph,
        occurances: Dict[str, int],
        qubit_mapping: Dict[str, int],
        n_qubits: int,
    ) -> QuantumCircuit:
        n = len(occurances)
        qc = QuantumCircuit(n_qubits)
        for i in range(n):
            if occurances["x"+str(i)] <= 1: continue
            elif occurances["x"+str(i)] == 2:
                theta_out = np.arccos(np.cos(self.theta)**(1./2.))
                qc.unitary(self.cloner_unitary(theta_out, theta_out), [qubit_mapping["y{}_1".format(i)], i])
            elif occurances["x"+str(i)] == 3:
                theta_mid = np.arccos(np.cos(self.theta)**(2./3.))
                theta_out = np.arccos(np.cos(self.theta)**(1./3.))
                qc.unitary(self.cloner_unitary(theta_out, theta_mid), [ qubit_mapping["y{}_1".format(i)], i])
                qc.unitary(self.cloner_unitary(theta_out, theta_out), [qubit_mapping["y{}_2".format(i)], qubit_mapping["y{}_1".format(i)]])
            else: raise Exception("cloning a qubit to >3 output qubits not yet implemented")
        return qc

class ExtendedVarNodeCloner(Cloner):
    """Equatorial (phase-covariant) cloner for variable nodes, now for arbitrary M outputs.

    Implements the 1→M optimal equatorial cloning network
    from Fan et al. (2001), using successive two-qubit unitaries.
    """

    def __init__(self, theta: float) -> None:
        """
        Args:
            theta: Bloch-sphere polar angle of the input state (|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩).
        """
        self.theta = theta

    def mark_angles(self, graph: nx.Graph, occurances: Dict[str, int]) -> None:
        """
        Mark each output (leaf) node l = 'y{i}_{k}' with its required rotation angle
        θ_out = arccos(cos(θ)^(1/M)), so that downstream controlled rotations know their targets.
        (Fan et al. 2001)
        """
        for node, data in graph.nodes(data=True):
            if data.get("type") == "output":
                # extract variable index i and total M = occurances["x{i}"]
                var = node.split("_")[0].replace("y", "x")
                M = occurances[var]
                theta_out = np.arccos(np.cos(self.theta) ** (1.0 / M))
                data["angle"] = [(theta_out, {})]

    def cloner_unitary(self, t_out: float, t_prev: float) -> np.ndarray:
        """
        Construct the 4×4 two-qubit cloning unitary that maps one clone + one blank to two clones,
        parameterized by (t_out, t_prev).  Reduces to Bužek–Hillery 1→2 when M=2.
        See Bužek & Hillery (1996); Fan et al. (2001).
        """
        aplus = (1/np.sqrt(2)) * (np.cos(0.5*(t_out - t_prev)) + np.cos(0.5*(t_out + t_prev))) \
                / np.sqrt(1 + np.cos(t_out)*np.cos(t_prev))
        amin  = (1/np.sqrt(2)) * (np.cos(0.5*(t_out - t_prev)) - np.cos(0.5*(t_out + t_prev))) \
                / np.sqrt(1 + np.cos(t_out)*np.cos(t_prev))
        bplus = (1/np.sqrt(2)) * (np.sin(0.5*(t_out + t_prev)) - np.sin(0.5*(t_out - t_prev))) \
                / np.sqrt(1 - np.cos(t_out)*np.cos(t_prev))
        bmin  = (1/np.sqrt(2)) * (np.sin(0.5*(t_out + t_prev)) + np.sin(0.5*(t_out - t_prev))) \
                / np.sqrt(1 - np.cos(t_out)*np.cos(t_prev))

        U = np.array([
            [aplus,   0.0,    0.0,    amin],
            [amin,    0.0,    0.0,   -aplus],
            [0.0,     bplus,  bmin,   0.0],
            [0.0,     bmin,  -bplus,  0.0],
        ]).T
        return U

    def generate_cloner_circuit(
        self,
        graph: nx.Graph,
        occurances: Dict[str, int],
        qubit_mapping: Dict[str, int],
        n_qubits: int,
    ) -> QuantumCircuit:
        """
        Build a Qiskit circuit that for each variable x_i with M>1 outputs,
        sequentially applies M-1 two-qubit cloning unitaries.

        - Compute the list of angles:
            θ_k = arccos(cos(θ)^(k / M)),  k=1,…,M-1   (Fan et al. 2001)
        - Apply U(t_k, t_{k-1}) between the (k-1)-th clone and the k-th blank.
        """
        qc = QuantumCircuit(n_qubits)
        for i in range(len(occurances)):
            var = f"x{i}"
            M = occurances.get(var, 1)
            if M <= 1:
                continue  # no cloning needed

            # 1→M equatorial cloning angles (symmetric subspace projection; Werner 1998)
            thetas = [np.arccos(np.cos(self.theta) ** (k / M)) for k in range(1, M)]
            
            # original qubit index in Qiskit register
            last = i
            for k, theta_k in enumerate(thetas, start=1):
                theta_prev = thetas[k-2] if k > 1 else theta_k
                # map to the k-th output qubit y{i}_{k}
                out_label = f"y{i}_{k}"
                target = qubit_mapping[out_label]
                # apply the two-qubit cloning unitary
                qc.unitary(self.cloner_unitary(theta_k, theta_prev), [target, last])
                last = target

        return qc
    
class OptimalCloner(Cloner):
    """Cloner optimized for two copies."""

    def __init__(self, theta: float, theta_marked: float) -> None:
        self.theta = theta
        self.theta_marked = theta_marked

    def mark_angles(self, graph: nx.Graph, occurances: Dict[str, int]) -> None:
        leaves = [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]
        for l in leaves:
            n = occurances[l.split("_")[0].replace("y","x")]
            if n == 1: graph.nodes[l]["angle"] = [(self.theta,{})]
            elif n == 2: graph.nodes[l]["angle"] = [(self.theta_marked,{})]
            else: raise Exception("cloning a qubit to >2 output qubits not yet implemented")

    def cloner_unitary(self) -> np.ndarray:
        # https://arxiv.org/pdf/quant-ph/9705038.pdf
        theta_paper = np.pi / 4. - 0.5*self.theta
        P = 0.5 * np.sqrt(1. + np.sin(2*theta_paper)) / np.sqrt(1 + np.sin(2*theta_paper)**2)
        Q = 0.5 * np.sqrt(1. - np.sin(2*theta_paper)) / np.sqrt(1 - np.sin(2*theta_paper)**2)
        a = (1. / np.cos(2*theta_paper)) * ( np.cos(theta_paper) * (P + Q*np.cos(2*theta_paper)) - np.sin(theta_paper) * (P - Q*np.cos(2*theta_paper)) )
        b = (1. / np.cos(2*theta_paper)) * P * np.sin(2*theta_paper) * (np.cos(theta_paper) - np.sin(theta_paper))
        c = (1. / np.cos(2*theta_paper)) * ( np.cos(theta_paper) * (P - Q*np.cos(2*theta_paper)) - np.sin(theta_paper) * (P + Q*np.cos(2*theta_paper)) )

        # gram schmidt
        u1 = np.array([a,b,b,c])
        u2 = np.array([c,b,b,a])
        v3 = np.random.uniform(size=(4))
        v3 = v3 / np.linalg.norm(v3)
        v4 = np.random.uniform(size=(4))
        v4 = v4 / np.linalg.norm(v4)

        u3 = v3 - np.inner(u1,v3)*u1 - np.inner(u2,v3)*u2
        u3 = u3 / np.linalg.norm(u3)
        u4 = v4 - np.inner(u1,v4)*u1 - np.inner(u2,v4)*u2 - np.inner(u3,v4)*u3
        u4 = u4 / np.linalg.norm(u4)

        # build unitary
        U =  np.array([
            [a, u3[0], c, u4[0]],
            [b, u3[1], b, u4[1]],
            [b, u3[2], b, u4[2]],
            [c, u3[3], a, u4[3]],
            ])

        # add some rotations to correct for different parametrisation
        def Ry(angle):
                return np.array([
                [np.cos(0.5*angle), -np.sin(0.5*angle)],
                [np.sin(0.5*angle),  np.cos(0.5*angle)]
            ])  

        return np.kron(Ry(-np.pi/2.), Ry(-np.pi/2.)) @ U @ np.kron(Ry(np.pi/2.), np.eye(2))

    def generate_cloner_circuit(
        self,
        graph: nx.Graph,
        occurances: Dict[str, int],
        qubit_mapping: Dict[str, int],
        n_qubits: int,
    ) -> QuantumCircuit:
        n = len(occurances)
        qc = QuantumCircuit(n_qubits)
        for i in range(n):
            if occurances["x"+str(i)] <= 1: continue
            elif occurances["x"+str(i)] == 2:
                qc.unitary(self.cloner_unitary(), [qubit_mapping["y{}_1".format(i)], i])
            else: raise Exception("cloning a qubit to >2 output qubits not yet implemented")
        return qc



class AsymmetricVarNodeCloner(Cloner):
    """Cloner favoring one copy depending on the distance to ``root``."""

    def __init__(self, theta: float, frac: float, root: int) -> None:
        self.theta = theta
        assert 0.0 <= frac <= 1.0
        self.frac = frac
        self.root = root

    def mark_angles(self, graph: nx.Graph, occurances: Dict[str, int]) -> None:
        leaves = [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]
        distances = dict(nx.all_pairs_shortest_path_length(graph))
        for l in leaves:
            n = occurances[l.split("_")[0].replace("y","x")]
            if n == 1:
                t = self.theta
            elif n == 2:
                # find out whether we are close or far away
                ours = l
                d_ours = distances["x{}_0".format(self.root)][ours]
                other = l.split("_")[0]+"_0" if int(l.split("_")[1])==1 else l.split("_")[0]+"_1"
                d_other = distances["x{}_0".format(self.root)][other]
                if d_ours < d_other: f = self.frac
                elif d_ours > d_other: f = 1. - self.frac
                else: f = 0.5 

                graph.nodes[l]["frac"] = f
                t = np.arccos(np.cos(self.theta)**f)
            else:
                raise Exception("cloning a qubit to >2 output qubits not yet implemented")
            graph.nodes[l]["angle"] = [(t,{})]

    def cloner_unitary(self, t1: float, t2: float) -> np.ndarray:
        aplus = (1./np.sqrt(2.)) * (np.cos(0.5*(t1-t2)) + np.cos(0.5*(t1+t2))) \
                     / np.sqrt(1. + np.cos(t1)*np.cos(t2))
        amin  = (1./np.sqrt(2.)) * (np.cos(0.5*(t1-t2)) - np.cos(0.5*(t1+t2))) \
                     / np.sqrt(1. + np.cos(t1)*np.cos(t2))
        bplus = (1./np.sqrt(2.)) * (np.sin(0.5*(t1+t2)) - np.sin(0.5*(t1-t2))) \
                     / np.sqrt(1. - np.cos(t1)*np.cos(t2))
        bmin  = (1./np.sqrt(2.)) * (np.sin(0.5*(t1+t2)) + np.sin(0.5*(t1-t2))) \
                     / np.sqrt(1. - np.cos(t1)*np.cos(t2))
        return np.array([
            [aplus, 0.,    0.,     amin   ],
            [amin,  0.,    0.,     -aplus ],
            [0.,    bplus, bmin,   0.     ],
            [0.,    bmin,  -bplus, 0.     ]
        ]).T

    def generate_cloner_circuit(
        self,
        graph: nx.Graph,
        occurances: Dict[str, int],
        qubit_mapping: Dict[str, int],
        n_qubits: int,
    ) -> QuantumCircuit:
        n = len(occurances)
        qc = QuantumCircuit(n_qubits)
        for i in range(n):
            if occurances["x"+str(i)] <= 1: continue
            elif occurances["x"+str(i)] == 2:
                f0 = graph.nodes["y"+str(i)+"_0"]["frac"]
                f1 = graph.nodes["y"+str(i)+"_1"]["frac"]
                print('Variable',i,': 0 has fraction', f0, 'and 1 has fraction', f1)
                t0 = np.arccos( np.cos(self.theta) ** f0 )
                t1 = np.arccos( np.cos(self.theta) ** f1 )
                qc.unitary(self.cloner_unitary(t0, t1), [qubit_mapping["y{}_1".format(i)], i])
            else: raise Exception("cloning a qubit to >2 output qubits not yet implemented")
        return qc


