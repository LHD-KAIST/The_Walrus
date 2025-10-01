# This code show us how to apply rotation operator
import strawberryfields as sf
from strawberryfields.ops import *

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D # contains tools  for creating 3D plots with Matplotlib

prog = sf.Program(1)

with prog.context as q:
    Fock(0) | q[0]
    Dgate(2) | q[0]
    Rgate(np.pi) | q[0]

eng = sf.Engine('fock', backend_options={"cutoff_dim": 15})
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

wigner_output_path = output_dir / "rotation_of_coherent_state.png"
fig.savefig(wigner_output_path, dpi=300, bbox_inches='tight')
plt.close(fig)
