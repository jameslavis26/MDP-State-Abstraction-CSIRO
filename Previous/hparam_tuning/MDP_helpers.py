import pandas as pd
import numpy as np

class MDP:
    def __init__(self, P, R, gamma=0.999, parent = None):
        self.transitions = P # P(A, S, S`)
        self.rewards = R # R(A, S, S`) or R(S, A)
        self.gamma = gamma

        self.N_actions = P.shape[0]
        self.N_states = P.shape[1]

        self.optimal_values = None
        self.optimal_policy = None

        self.k_states = None
        self.K = None

    def solve_MDP(self, N_iter=3000):
        self.optimal_values, self.optimal_policy = valueIteration(self.transitions, self.rewards, N_iter=N_iter)
    
    def check(self):
        mdptoolbox.util.check(self.transitions, self.rewards)

def valueFunction(mdp, policy, epsilon=1e-3, N_iter = 1e6):
    """ Calculate the value function of an mdp given a policy """
    S, A = mdp.N_states, mdp.N_actions
    gamma = mdp.gamma

    states = np.arange(S)

    R = np.zeros(S)
    for s in range(S):
        R[s] = mdp.rewards[s, policy[s]]

    T = np.zeros((S, S))
    for s in range(S):
        for sp in range(S):
            T[s, sp] = mdp.transitions[policy[s], s, sp]

    V = np.zeros(S)
    count = 0

    converged=False
    while not converged:
        V_new = R + gamma*np.matmul(T, V)

        if np.max(V_new - V) < epsilon:
            converged = True

        V = np.array(V_new)

        count += 1
        if count >= N_iter:
            print("Did not converge")
            break
        
    return V_new

def valueIteration(T, R, gamma=0.99, epsilon=1e-4, N_iter=10000):
    N_states, N_actions = R.shape

    V_new = np.zeros((N_states, 1))
    policy = np.zeros(N_states)

    A = np.matrix(np.eye(N_actions))

    R = np.matrix(R)

    count = 0
    while True:
        V_old = 1*V_new
        Q = 1*R
        for a in range(N_actions):
            Q += gamma*np.matmul(T[a, :, :], np.matmul(V_old, A[a, :]))
        policy = Q.argmax(axis=1)
        V_new = np.matrix(Q.max(axis=1))

        count += 1
        if count >= N_iter:
            # break
            raise Exception("Did not converge in time. Consider increasing the number of iterations.")

        if np.all(np.abs(V_new - V_old) <= epsilon):
            break
    return np.array(V_new).flatten(), np.array(policy).flatten()


def tabular_ds(mdp):
    """ Turn mdp into tabular dataset, returning X and y values. """
    dataset = mdp.transitions[0]
    for a in range(1, mdp.N_actions):
        dataset = np.concatenate([dataset, mdp.transitions[a]], axis=1)

    dataset = np.concatenate([dataset, mdp.rewards], axis=1)
    return dataset

def import_mdp(filepath="../datasets/reserve_729_6_K_8_feature_vector.csv"):
    mdp_df = pd.read_csv(filepath)
    N_states = mdp_df.shape[0]
    N_actions = len(mdp_df.columns[mdp_df.columns.map(lambda x: x[0] == 'R')])

    transitions = np.zeros((N_actions, N_states, N_states))
    for a in range(N_actions):
        cols = mdp_df.columns[mdp_df.columns.map(lambda x: x[0]=="P" and x[-1]==str(a+1))]
        transitions[a, :, :] = mdp_df.loc[:, cols].values

    rewards = np.zeros((N_states, N_actions))
    cols = mdp_df.columns[mdp_df.columns.map(lambda x: x[0]=="R")]
    rewards = mdp_df.loc[:, cols].values

    return MDP(transitions, rewards)

# def relabel_k(predicted_k_states):
#     """ Re number k states when a value is missing """
#     unique_k = np.unique(predicted_k_states)
#     new_K = len(unique_k)
#     new_k_states = np.zeros(len(predicted_k_states), dtype=int)

#     for i in range(new_K):
#         indices = np.where(predicted_k_states == unique_k[i])
#         new_k_states[indices] = i

#     return new_k_states

def relabel_k(predicted_k_states, K):
    """ Shift the value of the list until all labels are sequential """
    num = 0
    max_num = 1*K

    while True:
        if num not in predicted_k_states:
            predicted_k_states[predicted_k_states > num] -= 1
            max_num -= 1

            if num >= max_num:
                break
        else:
            num += 1
    return predicted_k_states
    