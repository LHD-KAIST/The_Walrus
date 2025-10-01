import strawberryfields as sf
from strawberryfields.ops import *
import numpy as np

prog = sf.Program(3) # how many modes do you want? For sf.Program(#), # is the number of modes.

with prog.context as q: 
    # operations(args1, args2, ...) | (q[0], q[1], q[2], ...)
    # State Preparation
    Vacuum() | q[0] # prepare a vacuum state
    Fock(1) | q[0]  # Fock(#) gives which fock states you wanna use |#>. 
    Ket((1,2,3)) | q[0] # prepare a ket vector represented in Fock basis. Check point: What happens do we violate the size of array?
    Coherent(1, np.pi) | q[0] # prepare a Coherent state(#, $) which has alpha = |#| * exp(i$)
    Squeezed(1, np.pi) | q[0] # prepare a Squeezed state(#, $) which has z = |#| * exp(i$)
    # prepare a Displaced Squeezed state(#, $, %, &) which has alpha = |#| * exp(i$), z = |%| * exp(i&)
    DisplacedSqueezed(1, np.pi, 2, 2*np.pi) | q[0]
    Thermal(1) | q[0] # prepare a Thermal state(#) which # is mean thermal population of the mode.
    #--------------Not yet Studied------------------------------------
    # Catstate, GKP, DensityMatrix, Gaussian
    #-----------------------------------------------------------------

    # Single-mode gates 
    Dgate(1, np.pi) | q[0] # phase space displacement gate.
    Xgate(1) | q[0] # Position displacement gate.
    Zgate(1) | q[0] # Momentum displacement gate.
    Sgate(1, np.pi) | q[0] # Phase space squeezing gate. 
    Rgate(np.pi) | q[0] # rotation gate in phase space
    Pgate(1) | q[0] # quadratic phase gate (shears the x quadrature)
    Vgate(0.1) | q[0] # cubic phase gate for non-Gaussian resources
    Fouriergate() | q[0] # Fourier transform that swaps x and p quadratures

    # Two-mode gates
    BSgate(np.pi / 4, np.pi / 2) | (q[0], q[1]) # beamsplitter mixing two modes
    MZgate(np.pi / 3, np.pi / 4) | (q[0], q[1]) # Mach-Zehnder interferometer built from two beamsplitters
    S2gate(0.35, np.pi / 2) | (q[1], q[2]) # two-mode squeezing gate
    CXgate(0.2) | (q[0], q[1]) # controlled addition gate in the position basis
    CZgate(0.2) | (q[0], q[1]) # controlled phase gate in the position basis
    CKgate(0.05) | (q[1], q[2]) # cross-Kerr interaction between modes

    # Additional state preparation operations
    Catstate(1.0, np.pi / 3, p=0.6) | q[2] # cat state superposition of coherent states
    GKP(state="0", epsilon=0.1) | q[2] # finite-energy Gottesman-Kitaev-Preskill state
    DensityMatrix(np.diag([0.6, 0.4])) | q[2] # prepare a mixed state specified by a density matrix
    Gaussian(np.identity(2), r=np.zeros(2)) | q[2] # prepare a general single-mode Gaussian state

    # Channels
    LossChannel(0.9) | q[0] # pure loss channel with transmissivity 0.9
    ThermalLossChannel(0.85, 0.1) | q[1] # thermal loss channel with transmissivity 0.85 and mean photons 0.1
    MSgate(0.4) | (q[0], q[2]) # measurement-based squeezing gate acting as a channel
    PassiveChannel(np.eye(2)) | (q[0], q[1]) # passive linear-optics channel defined by matrix T

    # Decompositions and Gaussian transformations
    Interferometer(np.eye(3)) | (q[0], q[1], q[2]) # apply a multi-mode interferometer defined by unitary U
    GraphEmbed(np.array([[0.0, 0.2], [0.2, 0.0]]), mean_photon_per_mode=0.3) | (q[0], q[1]) # embed a simple graph into an interferometer
    BipartiteGraphEmbed(np.array([[0.0, 0.5], [0.5, 0.0]])) | (q[0], q[1]) # embed a bipartite graph adjacency matrix
    GaussianTransform(np.eye(6)) | (q[0], q[1], q[2]) # apply a general Gaussian symplectic transformation

    # Measurements
    MeasureFock() | q[0] # photon counting measurement in the Fock basis
    MeasureThreshold() | (q[0], q[1]) # threshold detector distinguishing vacuum vs non-vacuum
    MeasureHomodyne(0.0) | q[0] # homodyne measurement of the x quadrature at angle 0
    MeasureHeterodyne() | q[0] # heterodyne measurement returning a complex amplitude
    MeasureX | q[0] # shorthand for homodyne measurement of the x quadrature
    MeasureP | q[0] # shorthand for homodyne measurement of the p quadrature
    MeasureHD | q[0] # shorthand for heterodyne measurement (outputs complex amplitude)


