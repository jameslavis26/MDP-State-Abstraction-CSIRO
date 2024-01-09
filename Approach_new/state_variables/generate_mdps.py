from reserve import generate_reserve
from kmdp_toolbox import aStarAbs
import numpy as np
from tqdm import tqdm 
import json
import os
import shutil

def valueIteration(T, R, gamma=0.99, epsilon=1e-4, N_iter=3000):
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

def generate_datsets(N_sites, N_species, K, N_datasets, path='datasets', folder = None, remove_previous=False):
    N_states = 3**N_sites
    N_actions = N_sites

    if not folder:
        folder = f'mdp_{N_states}_state'

    savefolder = os.path.join(path, folder, 'raw')
    folder_exists = os.path.isdir(savefolder)

    if not remove_previous and folder_exists:
        return "Data already exists"

    if remove_previous and folder_exists:
        print("Deleting folder ", savefolder)
        shutil.rmtree(savefolder)

    if not os.path.isdir(os.path.join(path, folder)):
        print("Creating folder ", savefolder)
        os.mkdir(os.path.join(path, folder))
        os.mkdir(savefolder)
        os.mkdir(os.path.join(path, folder, 'processed'))

    elif not os.path.isdir(savefolder):
        os.mkdir(savefolder)

    gamma=0.99

    print(f"Generating {N_datasets} MDPs with {N_states} states and {N_actions} actions \n")

    dataset = []
    for i in tqdm(range(N_datasets)):
        pj = np.random.random() # random probability between 0 and 1
        seed = i*(2+int(pj*100))

        P, R, state_variables = generate_reserve(N_sites, N_species, pj=pj, seed=seed) 

        optimal_values, optimal_policy = valueIteration(P, R)

        k_states, K = aStarAbs(P, R, optimal_values, optimal_policy, K=K, precision=1e-6)

        data = {
            "starting_probability": pj,
            "random_seed": seed,
            "state_variables": state_variables.tolist(),
            "transitions":P.tolist(),
            "rewards":R.tolist(),
            "optimal_values":optimal_values.tolist(),
            "k_states": k_states.tolist()
        }

        savefile = os.path.join(savefolder, f"mdp_{i}.json")

        file = open(savefile, "w")
        json.dump(data, file)
        file.close()



    