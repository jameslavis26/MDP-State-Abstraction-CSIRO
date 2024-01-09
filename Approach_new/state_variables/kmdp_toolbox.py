import numpy as np

def sk_to_s(L, K):
    """
    Maps values in `L` to ground states.

    Parameters:
    L (list): A list of integers representing the mapping of ground states to abstract states.
    K (int): The number of abstract states.

    Returns:
    dict: A dictionary where each key is an abstract state (0 to `K` - 1) and the corresponding value is a list of ground states that belong to that abstract state.

    Example:
    >>> L = [1, 1, 1, 2, 3, 4, 5, 5, 5, 5]
    >>> sk_to_s(L, 6)
    {1: [0, 1, 2], 2: [3], 3: [4], 5: [5, 6, 7, 8]}
    """
    return {k: [i for i, l in enumerate(L) if l == k] for k in range(K)}

def phi(s, S2K):
    return S2K[s]

def phi_inverse(sk, K2S):
    return K2S[sk]

# Assume R is always R(s,a)

def buildKMDP(P, R, L, K):
    """
    This function builds a K-abstract Markov Decision Process (MDP) from a ground MDP.
    
    Parameters
    ----------
    P : numpy.ndarray
        A 3D numpy array representing the transition probability matrix of the ground MDP.
        The shape of the array is (NA, NS, NS), where NA is the number of actions,
        and NS is the number of states in the ground MDP.
    R : numpy.ndarray
        A 3D numpy array representing the reward matrix of the ground MDP.
        The shape of the array is (K, NA, NS), where K is the number of possible rewards,
        NA is the number of actions, and NS is the number of states in the ground MDP.
    L : list
        A list of integers that maps each state in the ground MDP to an abstract state.
        Each integer in the list represents the abstract state that the corresponding ground state belongs to.
    K : int
        The number of abstract states in the K-MDP.
        
    Returns
    -------
    PK : numpy.ndarray
        A 3D numpy array representing the transition probability matrix of the K-MDP.
        The shape of the array is (NA, K, K).
    RK : numpy.ndarray
        A 2D numpy array representing the reward matrix of the K-MDP.
        The shape of the array is (K, NA).
    K2S : list
        A list of lists that maps each abstract state to a set of ground states.
        The length of the list is K, where K is the number of abstract states.
        Each sublist represents the set of ground states that belong to the corresponding abstract state.
    """    

    NS, NA = P.shape[2], P.shape[0] # Number of original states and number of actions on ground MDP
    
    K2S = sk_to_s(L,K)
    
    # Initialize abstract transition and rewards matrices
    PK = np.zeros((NA, K, K))
    RK = np.zeros((K, NA))
    
    #Initialize weights to 0
    w = np.zeros(NS)

    # Compute weights
    for sk in range(0,K,1):
        w[K2S[sk]] = 1 / len(K2S[sk])

        
    # Compute abstract reward matrix
    for k in range(0,K,1):
        for a in range(0, NA, 1):
            RK[k,a] = np.sum(R[phi_inverse(k,K2S),a]*w[phi_inverse(k, K2S)])
            
    # Compute abstract transition matrix
    
    for k1 in range(0,K,1):
        for k2 in range(0,K,1):
            for a in range(0,NA,1):
                S = K2S[k1]
                Sp = K2S[k2]
                aux = 0
                for x in range(0,len(S),1):
                    for y in range(0, len(Sp), 1):
                        aux = aux + P[a,S[x],Sp[y]]*w[S[x]]
                PK[a,k1,k2] = aux
                
    return PK, RK, K2S

def aStarAbs(P, R, V, policy, K, precision = 0.00001):
    
    """
    aStaR Algorithm for KMDP (K-state Markov Decision Process)
    
    This function implements the A* algorithm for KMDP, which is an algorithm to simplify a Markov Decision Process
    by abstracting its state space into K states.
    
    Parameters:
    - P (np.ndarray): 3-dimensional array (|S| x |S| x |A|) representing the transition probabilities of the original MDP
    - R (np.ndarray): 2-dimensional array (|S| x |A|) representing the rewards of the original MDP
    - V (np.ndarray): 1-dimensional array (|S|) representing the value function of the original MDP
    - policy (np.ndarray): 1-dimensional array (|S|) representing the policy of the original MDP
    - K (int): the number of states to which the state space of the original MDP should be abstracted
    - precision (float): the precision to be used in the binary search for the optimal d value. Default value is 0.00001.
    
    Returns:
    A tuple containing three arrays:
    - PK (np.ndarray): the transition probabilities of the abstracted MDP
    - RK (np.ndarray): the rewards of the abstracted MDP
    - K2S (np.ndarray): mapping from states in the abstracted MDP to states in the original MDP
    
    Example:
    ```
    PK, RK, K2S = aStarKMDP(P, R, V, Policy, K, precision = 0.00001)
    ```
    """
    
    NS, NA = P.shape[2], P.shape[0] # Number of original states and number of actions on ground MDP
    VMAX = V.max() #Maximum value
    d_lower, d_upper = 0.0, VMAX # Initialize d_lower to 0 and d_upper to VMAX
    p = d_upper - d_lower # Initialize the gap between d_upper and d_lower.
    L = np.zeros(NS)

    
    bindings = np.zeros((NS, 2)) # Initialize an array to store the bindings of states and policies
    
    number_abstractions = K + 1 # Keep track of the number of abstractions
    
    while p > precision:
        
        p = d_upper - d_lower; #To refine precision  
        d = d_lower + (d_upper - d_lower)/2; #Binary search on d
        
        bindings[:,0] = np.ceil(V/d) # Compute the bindings of states and policies
        bindings[:,1] = policy
        
        B, ia, ic = np.unique(bindings,return_index=True,return_inverse=True, axis=0)
        tmp_number_abstractions = len(ia)
        
        # Update the number of abstractions and the bounds of d
        if tmp_number_abstractions <= K:
            number_abstractions = tmp_number_abstractions
            L = ic
            d_upper = d
        else:
            d_lower = d
    
    if number_abstractions > K:
        raise ValueError("Number of abstractions obtained by the algorithm is higher than K")
        
    # PK, RK, K2S = buildKMDP(P, R, L, K) # Build the KMDP
    
    # return PK, RK, K2S
    return L, number_abstractions