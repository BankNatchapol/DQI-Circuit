import numpy as np
"""
quantize x between -C and C with B bits of accuracy
"""
def quantize(x, B, C=1.):
    x = np.float128(x)
    return np.float128(2.*C)/(np.float128(2)**B-np.float128(1.)) * np.floor(np.float128(0.5) + x*(np.float128(2)**B-np.float128(1.))/np.float128(2.*C) )


# This simple tets tells us that the rounding seems to work adequately well until around B=64
#x = np.float128(0.23456789)
#for B in range(128):
#    q = quantize(x, B)
#    print(B, "\t", q, "\t", np.abs(q-x))
#print(quantize(x, 6).dtype)


"""
rho is 2-qubit partial trace
idx is index of system to be traced out, can be 0 or 1
"""
def partial_trace(rho, idx):
    rho_tensor = rho.reshape([2, 2, 2, 2])
    return np.trace(rho_tensor, axis1=0+idx, axis2=2+idx)


I = np.array([[1.,0.],[0.,1.]])
X = np.array([[0.,1.],[1.,0.]])
H = np.array([[1.,1.],[1.,-1.]]) / np.sqrt(2.)
P0 = np.array([[1.,0.],[0.,0.]])
P1 = np.array([[0.,0.],[0.,1.]])
CNOT = np.kron(P0,I) + np.kron(P1,X)
NOTC = np.kron(I,P0) + np.kron(X,P1)
def Ry(t): return np.array([[np.cos(0.5*t), -np.sin(0.5*t)], [np.sin(0.5*t), np.cos(0.5*t)]])
def eqnode_unitary(alpha, beta): return CNOT @ np.kron(I,Ry(beta)) @ CNOT @ np.kron(X,Ry(alpha)) @ CNOT @ np.kron(X,I) @ NOTC


"""
Represents a single message. A message consists of a single data qubit and a discretised qubit angle coine.
The discretised angle consine is a B-bit number between -1 and 1 (using a fixed point number representation).
Both the qubit angle and the angle cosine are represented using np.float128 numbers.
"""
class Message(object):
    """
    create message with given qubit_state and angle_cosine.
    The angle_cosine is stored with B bits of accuracy.
    angle_cosine_ref is a reference without discretisation error.
    """
    def __init__(self, qubit_state, angle_cosine, angle_cosine_ref, B):
        assert -1. <= angle_cosine and angle_cosine <= 1.
        assert B >= 1
        B = int(B)
        self.B = B
        self.qubit_state = qubit_state
        self.angle_cosine = quantize(np.float128(angle_cosine), self.B)
        self.angle_cosine_ref = angle_cosine_ref

    """
    Performs an equality node operation on this message and the message `other`. Returns the output message.
    """
    def apply_eqnode_op(self, other):
        assert self.B == other.B
        t1 = np.arccos(self.angle_cosine)
        t2 = np.arccos(other.angle_cosine)
        a_min = (np.cos(0.5*(t1-t2)) - np.cos(0.5*(t1+t2)) ) / (np.sqrt(2.)*np.sqrt(1. + np.cos(t1)*np.cos(t2)))
        b_min = (np.sin(0.5*(t1+t2)) + np.sin(0.5*(t1-t2)) ) / (np.sqrt(2.)*np.sqrt(1. - np.cos(t1)*np.cos(t2)))
        alpha = np.arccos(-a_min) + np.arccos(-b_min)
        beta  = np.arccos(-a_min) - np.arccos(-b_min)
        if alpha > np.pi: alpha -= 2*np.pi
        if beta > np.pi: alpha -= 2*np.pi
        alpha = quantize(alpha, self.B+2, C=np.pi)
        beta = quantize(beta, self.B+2, C=np.pi)
        U = eqnode_unitary(alpha, beta)
        
        rho_tot = np.kron(self.qubit_state, other.qubit_state)
        rho_out = U @ rho_tot @ U.T
        rho_out =  partial_trace(rho_out, 1)
        angle_cosine_out = self.angle_cosine * other.angle_cosine
        angle_cosine_ref_out = self.angle_cosine_ref * other.angle_cosine_ref

        return Message(rho_out, angle_cosine_out, angle_cosine_ref_out, self.B)


    """
    Performs a check node operation on this message and the message `other`. Returns m1,m2,w1,w2 where
    m1, m2 are the possible output messages and w1, w2 are the weights of the two corresponding branches.
    """
    def apply_checknode_op(self, other):
        assert self.B == other.B
        rho_tot = np.kron(self.qubit_state, other.qubit_state)
        rho_out = CNOT @ rho_tot @ CNOT

        ket0 = np.array([[1.], [0.]], dtype=np.float128)
        ket1 = np.array([[0.], [1.]], dtype=np.float128)
        bra0 = np.array([[1., 0.]], dtype=np.float128)
        bra1 = np.array([[0., 1.]], dtype=np.float128)

        # branch 0:
        reduced = np.kron(I,bra0) @ rho_out @ np.kron(I,ket0)
        p0 = np.trace(reduced)
        rho_out_0 = reduced / p0
        angle_cosine_out_0 = (self.angle_cosine + other.angle_cosine) / (np.float128(1.) + self.angle_cosine*other.angle_cosine)
        angle_cosine_ref_out_0 = (self.angle_cosine_ref + other.angle_cosine_ref) / (np.float128(1.) + self.angle_cosine_ref*other.angle_cosine_ref)

        # branch 1:
        reduced = np.kron(I,bra1) @ rho_out @ np.kron(I,ket1)
        p1 = np.trace(reduced)
        rho_out_1 = reduced / p1
        angle_cosine_out_1 = (self.angle_cosine - other.angle_cosine) / (np.float128(1.) - self.angle_cosine*other.angle_cosine)
        angle_cosine_ref_out_1 = (self.angle_cosine_ref - other.angle_cosine_ref) / (np.float128(1.) - self.angle_cosine_ref*other.angle_cosine_ref)

        return Message(rho_out_0, angle_cosine_out_0, angle_cosine_ref_out_0, self.B), \
               Message(rho_out_1, angle_cosine_out_1, angle_cosine_ref_out_1, self.B), \
               p0, \
               p1

    """
    measures the data qubit of this message index idx in the +- basis.
    returns a numpy array of length 2 contatining the probabilities of obtaining a zero or one.
    """
    def measure(self):
        return np.trace(P0@H@self.qubit_state@H), np.trace(P1@H@self.qubit_state@H)

    def ideal_distinguishability(self):
        return 0.5*(1. + np.sin(np.arccos(self.angle_cosine_ref)))


"""
Stores the state of a system consisting of `n_qubits` messages. Basically, this is just a list of
length-`n_qubits` lists of `Message` objects plus a list of weights for each of these branches.
Equality and check node operations can be applied on a state. Each such operation acts on two
messages (indicated by their index) and the result is stored in the first of the two indexes.
The second message is then set to `None`.
"""
class State(object):

    """
    Creates an initial state with n_qubits messages where each message is in the angle init_angle.
    `codeword` is a list/np.array of size `n_qubits` that contains only 0 and 1. Id describes the 
    state of the channel output.
    Use B bits to store the angle cosines.
    """
    def __init__(self, n_qubits, init_angle, codeword, B):
        self.n_qubits = n_qubits
        init_angle = np.float128(init_angle)
        angle_cosine = np.cos(init_angle)
        # we start of with a single branch
        branch = list()
        for i in range(n_qubits):
            state = np.array([[np.cos(0.5*init_angle)**2,                                       (-1)**codeword[i]*np.cos(0.5*init_angle)*np.sin(0.5*init_angle)],
                              [(-1)**codeword[i]*np.cos(0.5*init_angle)*np.sin(0.5*init_angle), np.sin(0.5*init_angle)**2]],
                             dtype=np.float128)
            branch.append(Message(state, angle_cosine, angle_cosine, B))
        self.branches = [branch]
        self.weights = [1.]

    """
    applies an equality node operation on the idx1-th and idx2-th messages and stores the result in the
    idx1-th message.
    """
    def apply_eqnode_op(self, idx1, idx2):
        assert 0 <= idx1 and idx1 < self.n_qubits
        assert 0 <= idx2 and idx2 < self.n_qubits

        for i in range(len(self.branches)):
            self.branches[i][idx1] = self.branches[i][idx1].apply_eqnode_op(self.branches[i][idx2])
            self.branches[i][idx2] = None 

    """
    applies a check node operation on the idx1-th and idx2-th messages and stores the result in the
    idx1-th message.
    """
    def apply_checknode_op(self, idx1, idx2):
        assert 0 <= idx1 and idx1 < self.n_qubits
        assert 0 <= idx2 and idx2 < self.n_qubits

        branches_old = self.branches
        weights_old = self.weights
        self.branches = list()
        self.weights = list()
        for i in range(len(branches_old)):
            b = branches_old[i]
            w = weights_old[i]
            b1 = b.copy()
            b2 = b.copy()
            m1, m2, w1, w2 = b[idx1].apply_checknode_op(b[idx2])
            b1[idx1] = m1
            b2[idx1] = m2
            b1[idx2] = None
            b2[idx2] = None
            self.branches.append(b1)
            self.weights.append(w*w1)
            self.branches.append(b2)
            self.weights.append(w*w2)

    """
    measures the message qubit with index idx in the +- basis.
    returns a numpy array of length 2 contatining the probabilities of obtaining a zero or one.
    """
    def measure(self, idx):
        assert 0 <= idx and idx < self.n_qubits
        prob0 = np.float128(0.)
        prob1 = np.float128(0.)
        for i in range(len(self.branches)):
            p0,p1 = self.branches[i][idx].measure()
            prob0 += self.weights[i] * p0
            prob1 += self.weights[i] * p1
        return np.array([prob0, prob1])

    def ideal_distinguishability(self, idx):
        assert 0 <= idx and idx < self.n_qubits
        prob = np.float128(0.)
        for i in range(len(self.branches)):
            p = self.branches[i][idx].ideal_distinguishability()
            prob += self.weights[i] * p
        return prob

    """
    Print debug information about the state. Mostly useful for debugging/development.
    """
    def print_debug(self):
        n_branches = len(self.branches)
        print('Number of branches:', n_branches)
        for i in range(n_branches):
            print("="*10)
            print('Branch {} (weight={}):'.format(i, self.weights[i]))
            for j in range(self.n_qubits):
                if self.branches[i][j] is None: print("* None")
                else:
                    print("* probabs=",self.branches[i][j].measure(), "\tangle cosine=", self.branches[i][j].angle_cosine)
                    print(self.branches[i][j].qubit_state)
