import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np
from pathlib import Path

prog = sf.Program(3)

with prog.context as q:
    Fock(1) | q[0]
    Squeezed(2) | q[1]
    Coherent(1, np.pi) | q[2]


# more details about which backend is effective for your situation : https://strawberryfields.readthedocs.io/en/stable/introduction/circuits.html
eng = sf.Engine("fock", backend_options={"cutoff_dim" : 5})
#eng = sf.Engine("gaussian", backend_options={"cutoff_dim" : 5})
#eng = sf.Engine("bosonic", backend_options={"cutoff_dim" : 5})
#eng = sf.Engine("tf", backend_options={"cutoff_dim" : 5})

# Once the engine has been initialized, the quantum program can be executed on the selected backend 
result = eng.run(prog)

# Execution Result
print(result.state) # 
state = result.state
print(state.trace()) # trace
print(state.dm().shape) # density matrix
print(result.samples) # Measurement samples from any measurements performed.
