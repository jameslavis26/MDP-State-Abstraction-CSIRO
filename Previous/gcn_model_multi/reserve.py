import numpy as np

def getSite(state_id, J):
    baseTern = np.power(3*np.ones(J), np.arange(J))
    site = np.zeros(J)
    for i in reversed(range(J)):
        if state_id - 2*baseTern[i] >= 0:
            site[i] = 2
            state_id = state_id - 2*baseTern[i]
        elif state_id - baseTern[i] >=0:
            site[i] = 1
            state_id = state_id - baseTern[i]
    return site

def getSite(stateid, J):
    baseTern = np.power(np.tile(3, J), np.arange(J))
    site = np.zeros(J, dtype=int)
    for i in range(J-1, -1, -1):
        if stateid - 2*baseTern[i] >= 0:
            site[i] = 2
            stateid = stateid - 2*baseTern[i]
        elif stateid - baseTern[i] >= 0:
            site[i] = 1
            stateid = stateid - baseTern[i]
    return site

def getState(site):
    J = len(site)
    baseTern = np.power(np.tile(3, J), np.arange(J))
    state = np.sum(site * baseTern)
    return int(state)

def dec2binvec(dec, n=None):
    # Convert the decimal number to a binary string.
    if n is not None:
        out = np.binary_repr(dec, n)
    else:
        out = np.binary_repr(dec)
    
    # Convert the binary string to a binary vector.
    out = np.array([int(b) for b in out])
    
    return out

def binvec2dec(vec):
    # Non-zero values map to 1.
    vec = np.array([0 if v == 0 else 1 for v in vec])
    L = len(vec)

    h = sum([vec[L-i-1]*2**i for i in range(len(vec))])

    return h


def mdp_example_reserve(M, pj=0.1):
    """
    M(JxI) = species distribution across sites
    %        J = number of sites (> 0), optional (default 5)
    %        I = number of species (>0), optional (default 7)
    %   pj = probability of development occurence, in ]0, 1[, optional (default 0.1)
    """

    [J, I] = M.shape
    
    # Definition of states
    S = 3**J # For each site, 0 is available, 1 is reserved, 2 is developed
    A = J # J actions

    """
    % There are J actions corresponding to the selection of a site for
    % reservation. A site can only be reserved if it is available.
    % By convention we will use a ternary base where state #0 is the state
    % that corresponds to [0,0, ..,0] all sites are available. State #1 is
    % [1,0,0,...,0]; state 2 is [2,0,0 ..,0] and state 3 is [0,1,0, .. 0] and so forth.
    % for example 
    % site = [0,0,1,2] means the first 2 sites are available (site(1:2)=0), site 3 is
    % reserved (site(3)=1) and site 4 is developped (site(4)=2).
    """

    P = np.zeros(shape=(A, S, S))
    for s1 in range(S): # For all states
        site1 = getSite(s1, J)
        for a in range(A): # For all actions
            site2 = np.array(site1) # Site 1 after action A is done

            # If site1[a] is available, change to reserved
            if site1[a] == 0:
                site2[a] = 1

            is_available = site2 == 0
            n_available = sum(is_available)
            if n_available > 0:
                siten = np.ones((2**n_available, 1))*site2
                aux = np.zeros(n_available)

                for k in range(2**n_available):
                    siten[k, is_available] = aux*2
                    ndev = sum(abs(site2[is_available] - aux))
                    s2 = getState(siten[k, :])
                    P[a, s1, s2] = pj**ndev * (1-pj)**(n_available - ndev)
                    aux = dec2binvec(binvec2dec(aux)+1, n_available)
            
            else:
                s2 = getState(site2)
                P[a, s1, s2] = 1

    R = np.zeros((S, A))
    for s1 in range(S):
        site1 = getSite(s1, J)
        for a in range(J):
            if site1[a] != 2:
                targetSp = M[a, :]
                reservedSites = np.where(site1 == 1)[0]
                if reservedSites.size > 0:
                    reservedSp = np.max(M[reservedSites, :], axis=0)
                    R[s1, a] = np.sum(np.maximum(0.0001, targetSp-reservedSp))
                else:
                    R[s1, a] = np.sum(np.maximum(0.0001, targetSp))
                
    return P, R

def generate_reserve(J, I, pj=0.15, discount=0.96, seed=0):
    # J = 8 # Number of sites
    # I = 20 # number of species
    # pj = 0.15 # probability of development occuring

    discount = 0.96

    # generate species richness matrix
    np.random.seed(seed)
    M = np.random.random(size=(J, I))

    # Generate transitions and reward matrix
    P, R = mdp_example_reserve(M, pj)

    R = R - np.min(R, axis=0)
    R = R/np.max(R, axis=0)

    return P, R

# def main():
#     path = "./reserve_datasets"
#     N_sets = 10
#     for i in range(N_sets):
#         P, R = generate_reserve(8, 20)
