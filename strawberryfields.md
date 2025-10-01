History

In 2020 December, USTC Jian-Wei Pan reached quantum supremacy by implementing gaussian boson sampling on 76 photons with their photonic quantum computer Jiuzhang. -> upgrade to Jiuzhang 2.0 or others

CVQC -> photons
(1) DVQC vs CVQC

DVQC : Hilbert space,. Qubits, Basis = {$|0\rangle$, $|1\rangle$}
Assign mode(energy) to each particle
Then, simplest case will be 2 mode. each mode can have only 1 particles only. We can present general state as $| \psi \rangle $ = $\alpha |0\rangle$ + $\beta |1\rangle$ .
Physical realization is Transmon qubit, etc.
Can be done only under 20mK
There are lots of gates, hadamard, Cnot


CVQC : Fock space, Qumodes, Basis = {$|0\rangle$, $|1\rangle$, $|2\rangle$, ..., $|n\rangle$}
Assign particles to each mode
In this case, smallest(simplest) case will be 1 mode with n particles. 
Then, each mode can have infinite particles in principle. For 1 mode, Basis are such as : $|0\rangle$ = 0 particles, $|1 \rangle$ = 1 particles, $|2\rangle$ = 2 particles...
Then multiple modes is just like tensor product of each mode.
Physical realization is done by photons, room temperature
There are agtes, displacement, squeezing. 
Measurements is done by homodyne detection, heterodyne detection, photon counts. 