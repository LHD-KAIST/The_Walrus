import strawberryfields as sf
from strawberryfields.ops import *

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D # contains tools  for creating 3D plots with Matplotlib

# creates a quantum program with one mode or one quantum system.
prog = sf.Program(1)

# prepare a Fock state in the # mode of quantum system
with prog.context as q:
    Fock(0) | q[0] # Fock(#) means we wanna prepare # photons, and q(#) gives we will fill photons on (#-1)th mode

# This code runs the quantum program above on the engine using the Fock backend.
eng = sf.Engine('fock', backend_options={"cutoff_dim": 5})
state = eng.run(prog).state

# plot 3D surface plot of the Wigner function
output_dir = Path(__file__).resolve().parent / "plots"
output_dir.mkdir(exist_ok=True)

fig = plt.figure()
X = np.linspace(-5, 5, 100)
P = np.linspace(-5, 5, 100)
Z = state.wigner(0, X, P)
X, P = np.meshgrid(X, P)
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)

# set the viewing angle
ax.view_init(elev=15, azim=20)
fig.set_size_inches(4.8, 5)

wigner_output_path = output_dir / "fock_state_wigner_surface.png"
fig.savefig(wigner_output_path, dpi=300, bbox_inches='tight')
plt.close(fig)
