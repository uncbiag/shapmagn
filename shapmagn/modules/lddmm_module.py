from torch.autograd import grad


############## Hamiltonian view of LDDMM ######################3

def hamiltonian(K):
    def H(p, q):
        return .5 * (p * K(q, q, p)).sum()

    return H

def hamiltonian_system(K):
    H = hamiltonian(K)

    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp

    return HS


def hamiltonian_shooting(p0, q0, K, nt, integrator):
    return integrator(hamiltonian_system(K), (p0, q0), nt)




###################  variational view of LDDMM ######################




def variational_evolve(p0,q0,K,grad_K, nt, integrator):
    q0 + h * K(q0, q0, p0), \
    p0 - h * grad_K(p0, q0)