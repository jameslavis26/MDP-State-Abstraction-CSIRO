import torch
from kmdp_toolbox import sk_to_s

def buildKMDP(T: torch.tensor, R: torch.tensor, predicted_k_states: torch.tensor, K: int, device='cpu') -> torch.tensor:
    N_states, N_actions = R.shape

    """ Implement buildKMDP using inbuilt torch functions to keep everything on device """
    K2S = sk_to_s(predicted_k_states, K)
    weights = (1/torch.bincount(predicted_k_states))[predicted_k_states]

    RK = torch.empty(size=(K, N_actions), device=device, dtype=torch.float64)
    # R = torch.tensor(mdp.rewards).to(device)

    TK = torch.empty(size=(N_actions, K, K), device=device, dtype=torch.float64)
    # T = torch.tensor(mdp.transitions).to(device)

    for k in range(K):
        RK[k] = (R.T*weights).T[predicted_k_states==k].sum(axis=0)
        for kp in range(K):
            TK[:, k, kp] = (T[:, :, predicted_k_states==kp].sum(axis=2) * weights)[:, predicted_k_states==k].sum(axis=1)
    return TK, RK, K2S

# value iteration

def valueIteration(T: torch.tensor, R: torch.tensor, gamma = 0.99, epsilon=1e-4, N_iter=10000, device='cpu') -> torch.tensor:
    """ Implement Value Iteration in the pytorch environment """
    N_states, N_actions = R.shape
    V = torch.zeros(size=[N_states], device=device, dtype=torch.float64)
    Q = torch.empty(size=[N_states, N_actions], device=device, dtype=torch.float64)
    for i in range(N_iter):
        for a in range(N_actions):
            Q[:, a] = R[:, a].T + gamma*T[a, :, :]@V

        V_new, policy = Q.max(axis=1)

        if torch.all(torch.abs(V_new - V) < epsilon):
            break
        
        if i == N_iter - 1:
            raise Exception("Did not converge in time. Consider increasing the number of iterations.")

        V = V_new
    return V_new, policy

def valueFunction(T, R, policy, gamma=0.99, epsilon=1e-3, N_iter = 1e6, device='cpu'):
    """ Calculate the value function of an mdp given a policy """
    N_states, N_actions = R.shape
    
    V = torch.zeros(size=[N_states], device=device)
    V_new = torch.zeros(size=[N_states], device=device)

    count = 0
    converged=False
    while not converged:
        for s in range(N_states):
            V_new[s] = R[s, policy[s]] + gamma*(T[policy[s], s]*V).sum()

        if torch.max(V_new - V) < epsilon:
            converged = True

        V = 1*V_new

        count += 1
        if count >= N_iter:
            print("Did not converge")
            break
        
    return V_new

def calculate_gap(T, R, V, predicted_k_states, K, device='cpu'):
    T = T.to(device)
    R = R.to(device)
    V = V.to(device)
    predicted_k_states = predicted_k_states.to(device)

    N_states, N_actions = R.shape

    new_K = len(predicted_k_states.unique())
    predicted_k_states = relabel_k(predicted_k_states, K) if new_K != K else predicted_k_states

    PK, RK, K2S = buildKMDP(T, R, predicted_k_states, new_K, device=device)
    _, kmdp_policy = valueIteration(PK, RK, gamma=0.99, N_iter=50000, epsilon=1e-1, device=device)

    k_policy = torch.empty(size=[N_states], dtype=torch.int64, device=device)

    for k in range(new_K):
        k_policy[K2S[k]] = kmdp_policy[k]

    V_K = valueFunction(T, R, k_policy, device=device)

    gap = torch.max(torch.abs(V - V_K))
    error = gap/max(V)

    return gap, error


# ChatGPT written
def multiclass_recall_score(y_true, y_pred, average='macro'):
    """
    Calculate the multiclass recall score using PyTorch.

    Parameters:
    - y_true (torch.Tensor): True labels (ground truth).
    - y_pred (torch.Tensor): Predicted labels.
    - average (str): Type of averaging to use for multiclass recall.
        - 'macro' (default): Calculate recall for each class and then take the average.
        - 'micro': Calculate recall globally by considering all instances.
        - 'weighted': Calculate recall for each class and weight them by support.

    Returns:
    - recall (float): The multiclass recall score.
    """
    assert len(y_true) == len(y_pred), "Input arrays must have the same length"

    if average not in ('macro', 'micro', 'weighted'):
        raise ValueError("Invalid 'average' parameter. Use 'macro', 'micro', or 'weighted'.")

    num_classes = len(torch.unique(y_true))
    recall_per_class = []

    for class_label in range(num_classes):
        true_positive = torch.sum((y_true == class_label) & (y_pred == class_label)).item()
        false_negative = torch.sum((y_true == class_label) & (y_pred != class_label)).item()
        recall = true_positive / (true_positive + false_negative + 1e-10)  # Adding a small epsilon to avoid division by zero
        recall_per_class.append(recall)

    if average == 'macro':
        return sum(recall_per_class) / num_classes
    elif average == 'micro':
        total_true_positives = torch.sum((y_true == y_pred) & (y_true == class_label)).item()
        total_false_negatives = torch.sum((y_true != y_pred) & (y_true == class_label)).item()
        return total_true_positives / (total_true_positives + total_false_negatives + 1e-10)
    elif average == 'weighted':
        class_counts = [torch.sum(y_true == class_label).item() for class_label in range(num_classes)]
        total_samples = len(y_true)
        weights = [count / total_samples for count in class_counts]
        weighted_recall = sum([recall_per_class[i] * weights[i] for i in range(num_classes)])
        return weighted_recall
    

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
    