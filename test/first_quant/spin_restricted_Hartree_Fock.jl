md"""
## Construction the system

First make two hydrogen atoms spaced by $d = 2$.
"""

using WTP, GTO, Test, LinearAlgebra

d = 2.0
h2 = [make_atom(:H, 0.0, 0.0, 0.0), make_atom(:H, d, 0.0, 0.0)]

md"""
Given the molecule, we can compute the total number of 
electrons by summing over the nuclear charges.
"""

n_e = sum(charge, h2)

@test n_e == 2

md"""
## Pick a basis
Hartree-Fock is first based on the approximation

$$
\begin{equation}
\Psi = \psi_1 \wedge \psi_2 \wedge \ldots \psi_{N_e}.
\end{equation}
$$

This approximation is further approximated with 

$$
\psi_i(r) = \sum_j C_{j i} \chi_j(r).
$$

We can pick a basis set from basis set exchange 
[BSE](https://www.basissetexchange.org/) as follows
"""

basis_set = load_basis("6-21g.1.json")
basis = generate_basis(basis_set, h2...)
n_b = length(basis)
@test n_b == 4

md"""
## Evaluate the integrals

To evaluate the energy, we first need the one-electron integrals between the basis functions.

$$
\begin{align}
S_{p q} &= \int \chi_p^*(r) \chi_q(r) \mathrm{d} r,\\
T_{p q} &= \int \chi_p^*(r) \nabla^2 \chi_q(r) \mathrm{d} r,\\
A_{p q} &= \sum_i \int \frac{\chi_p^*(r) \chi_q(r)}{|r - R_i|} \mathrm{d} r.
\end{align}
$$

These integrals can be constructed as follows

"""

S = [p' * q for p in basis, q in basis]
T = [p' * ∇² * q for p in basis, q in basis]
A = [p' * VNuc(h2...) * q for p in basis, q in basis]

md"""
The more challenging part is the two electron integrals, which is 
a four-legged tensor 

$$
\begin{equation}
J_{p q r s} = V_{p s r q} = \iint \frac{\chi_p(r) \chi_q(r) \chi_r(r') \chi_s(r') }{|r - r'|} \mathrm{d}r \mathrm{d} r'
\end{equation}
$$

This can also be contructed through a comprehension and a `permutedims`.
"""

J = [(p * q | r * s) for p in basis, q in basis, r in basis, s in basis]
K = permutedims(J, (1, 3, 2, 4))


md"""
If we are doing RHF, we need the following tensors.
"""
h = T + A
Q = 2J - K


md"""
## Self consistent field iteration

The self-consistent field iteration involves iteratively  constructing and diagonalizing the Fock matrix. The construction of the Fock matrix is

$$
\begin{equation}
F(C)_{p q} = h_{p q} + \sum_{rs} Q_{p q r s} \sum_n C_{r n} C_{s n} 
\end{equation}
$$

"""
fock(C) = h + [dot(C, Q[p, q, :, :] * C) for p in 1:n_b, q in 1:n_b]

md"""
We can then diagonalize the Fock matrix and take the first $N_e / 2$
eigenvectors as the new $C$. The SCF update is thus
"""

scf_update(C, S) = eigen(Hermitian(fock(C)), Hermitian(S)).vectors[:, 1:div(n_e, 2)]

md"""
Importantly, we casted the Fock matrix and the overlap matrix 
to Hermitian to tell Julia's eigensolver to use the algorithm for Hermitian matrices, which is drastically better than the general one.

With all the routines in place, we can solve for the HF ground state of 
H2. First, make a wish to the random number generator
"""
using Random
Random.seed!(2077)
C = qr(rand(n_b, div(n_e, 2))).Q |> Matrix

md"""
Then, we can do the SCF iteration just by repeatedly applying the SCF update.
"""
for _ in 1:100
    C = scf_update(C, S)
end

md"""
## Energy evaluation

Don't forget to add the nuclear energy 

$$
\begin{equation}
E_{\mathrm{NN}} = \frac{1}{2} \sum_{i \neq j} \frac{Z_i Z_j}{|R_i - R_j|}
\end{equation}
$$
"""

e_nuc(atoms) = 1 / 2 * sum(charge(i) * charge(j) / norm(coordinates(i) - coordinates(j)) for i in atoms, j in atoms if j != i)

md"""
We can then compute the total energy as 

$$
\begin{equation}
E_{\mathrm{HF}} = \sum_{p q} (h + F(C))_{p q} \sum_n C_{q n} C_{p n} + E_{\mathrm{NN}}\label{eqn:etotal}
\end{equation}
$$
"""

energy(C) = real(dot(C, (h + fock(C)) * C)) + e_nuc(h2)

@test isapprox(energy(C), -1.0802700699226433)

