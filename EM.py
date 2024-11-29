import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Load dataset
data = pd.read_csv('traindata.txt', delim_whitespace=True, header=None, names=['X1', 'X2', 'X3', 'M', 'C'])

# Constants
NUM_PATIENTS = len(data)
MAX_ITER = 1000
DELTA_THRESHOLD = 0.01

# Priors
P_C = np.array([0.45, 0.2, 0.35])  # [C=0, C=1, C=2]
P_M = np.array([0.92, 0.08])       # [M=False, M=True]

P_X1_given_M_C = {
    True: np.array([[0.98, 0.02], [0.97, 0.03], [0.98, 0.02]]),
    False: np.array([[0.99, 0.01], [0.03, 0.97], [0.02, 0.98]])
}

P_X2_given_C = np.array([[0.8, 0.2], [0.1, 0.9], [0.99, 0.01]])
P_X3_given_C = np.array([[0.75, 0.25], [0.85, 0.15], [0.08, 0.92]])

# Add noise to priors
def add_noise(cpt, delta):
    noise = np.random.uniform(0, delta, size=cpt.shape)
    noisy_cpt = cpt + noise
    return normalize(noisy_cpt, axis=1, norm='l1')  # Ensure the probabilities sum to 1

# Initialize with noise
def initialize_with_noise(delta):
    noisy_CPTs = {
        "P_C": add_noise(P_C.reshape(1, -1), delta).flatten(),
        "P_M": add_noise(P_M.reshape(1, -1), delta).flatten(),
        "P_X1_given_M_C": {key: add_noise(val, delta) for key, val in P_X1_given_M_C.items()},
        "P_X2_given_C": add_noise(P_X2_given_C, delta),
        "P_X3_given_C": add_noise(P_X3_given_C, delta)
    }
    return noisy_CPTs

# Expectation step
def expectation(data, CPTs):
    weights = np.zeros((NUM_PATIENTS, 3))  # Weights for C=0, C=1, C=2
    for i, row in data.iterrows():
        x1, x2, x3, m, c = row
        if c != -1:  # If C is labeled (not -1)
            weights[i, :] = [1 if c == c_val else 0 for c_val in range(3)]  # Hard-assign weights
        else:  # If C is unlabeled
            for c_val in range(3):  # Compute posterior for all C=0, 1, 2
                p_c = CPTs["P_C"][c_val]
                p_x1 = CPTs["P_X1_given_M_C"][bool(m)][c_val, x1]
                p_x2 = CPTs["P_X2_given_C"][c_val, x2]
                p_x3 = CPTs["P_X3_given_C"][c_val, x3]
                weights[i, c_val] = p_c * p_x1 * p_x2 * p_x3
            weights[i] /= np.sum(weights[i])  # Normalize
    return weights

# Maximization step
def maximization(data, weights):
    # Update P(C)
    new_P_C = np.sum(weights, axis=0) / NUM_PATIENTS
    
    # Update P(M)
    new_P_M = np.array([
        np.sum(weights[data['M'] == m]) / NUM_PATIENTS
        for m in [False, True]
    ])
    
    # Update P(X1 | M, C)
    new_P_X1_given_M_C = {
        m: np.zeros_like(P_X1_given_M_C[m])
        for m in [True, False]
    }
    for m in [True, False]:
        m_mask = (data['M'] == m)
        for c_val in range(3):
            for x1_val in [0, 1]:
                mask = (data['X1'] == x1_val) & m_mask
                new_P_X1_given_M_C[m][c_val, x1_val] = np.sum(weights[mask, c_val])
    
    # Normalize P(X1 | M, C)
    for m in new_P_X1_given_M_C:
        new_P_X1_given_M_C[m] = normalize(new_P_X1_given_M_C[m], axis=1, norm='l1')

    # Update P(X2 | C) and P(X3 | C)
    new_P_X2_given_C = np.zeros_like(P_X2_given_C)
    new_P_X3_given_C = np.zeros_like(P_X3_given_C)
    for c_val in range(3):
        for x2_val in [0, 1]:
            new_P_X2_given_C[c_val, x2_val] = np.sum(weights[data['X2'] == x2_val, c_val])
        for x3_val in [0, 1]:
            new_P_X3_given_C[c_val, x3_val] = np.sum(weights[data['X3'] == x3_val, c_val])
    
    # Normalize P(X2 | C) and P(X3 | C)
    new_P_X2_given_C = normalize(new_P_X2_given_C, axis=1, norm='l1')
    new_P_X3_given_C = normalize(new_P_X3_given_C, axis=1, norm='l1')

    return {
        "P_C": new_P_C,
        "P_M": new_P_M,
        "P_X1_given_M_C": new_P_X1_given_M_C,
        "P_X2_given_C": new_P_X2_given_C,
        "P_X3_given_C": new_P_X3_given_C
    }

# EM algorithm
def em_algorithm(data, delta):
    CPTs = initialize_with_noise(delta)
    prev_likelihood = -np.inf
    for _ in range(MAX_ITER):
        weights = expectation(data, CPTs)
        CPTs = maximization(data, weights)
        likelihood = np.sum(np.log(np.sum(weights, axis=1)))
        if abs(likelihood - prev_likelihood) < DELTA_THRESHOLD:
            break
        prev_likelihood = likelihood
    return CPTs, likelihood

# Run experiments for varying delta
deltas = np.linspace(0, 4, 20)
results = []

for delta in deltas:
    for _ in range(20):  # 20 trials
        CPTs, likelihood = em_algorithm(data, delta)
        results.append((delta, likelihood))

# Save results to a DataFrame
results_df = pd.DataFrame(results, columns=["Delta", "Likelihood"])

# Save raw results to CSV
results_df.to_csv("em_likelihood_results.csv", index=False)

# Analyze results
results_summary = results_df.groupby("Delta").agg(
    Mean_Likelihood=('Likelihood', 'mean'),
    Std_Likelihood=('Likelihood', 'std')
)

# Save summary to CSV
results_summary.to_csv("em_likelihood_summary.csv")

# Print summary
print(results_summary)
