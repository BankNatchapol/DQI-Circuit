import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dqi.qiskit.decoding.BPQM.cirq.bpqm import *
from dqi.qiskit.decoding.BPQM.cirq.decoders import *
from dqi.qiskit.decoding.BPQM.cirq.linearcode import *
from dqi.qiskit.decoding.BPQM.cirq.cloner import *

print('Cirq BPQM test:')
# TODO: Add actual Cirq BPQM test logic here, e.g. run a simple BPQM circuit and print result 