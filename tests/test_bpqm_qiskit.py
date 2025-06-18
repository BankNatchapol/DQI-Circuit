import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dqi.qiskit.decoding.BPQM.qiskit.bpqm import *
from dqi.qiskit.decoding.BPQM.qiskit.decoders import *
from dqi.qiskit.decoding.BPQM.qiskit.linearcode import *
from dqi.qiskit.decoding.BPQM.qiskit.cloner import *

print('Qiskit BPQM test:')
# TODO: Add actual Qiskit BPQM test logic here, e.g. run a simple BPQM circuit and print result 