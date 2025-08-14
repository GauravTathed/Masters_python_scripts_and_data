import numpy as np
import itertools
from scipy.linalg import expm
from scipy.optimize import minimize
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- All helper functions defined at module level ---

dim = 9

def coupling_operator_with_phase(i, j, dim, phi):
    op = np.zeros((dim, dim), dtype=complex)
    op[i, j] = np.exp(1j * phi)
    op[j, i] = np.exp(-1j * phi)
    return op

def pulse_operator(coupling, f, phi):
    theta = np.pi * f      
    phi_rad = np.pi * phi  
    H = coupling_operator_with_phase(coupling[0], coupling[1], dim, phi_rad)
    U = expm(-1j * 0.5 * H * theta)
    return U

def canonicalize(coupling, phase_flag):
    i, j = coupling
    if i != 0 and j == 0:
        return (0, i), phase_flag + 1.0
    else:
        return coupling, phase_flag

def candidate_unitary(seq, x):
    U = np.eye(dim, dtype=complex)
    k = len(seq)
    for i in range(k):
        f = x[2*i]
        phi = 0 if x[2*i+1] < 0.5 else 1
        U_i = pulse_operator(seq[i], f, phi)
        U = U_i @ U
    return U

def optimal_global_phase(U, U_target):
    theta_opt = np.angle(np.trace(U @ np.conjugate(U_target).T))
    return theta_opt

def cost_function(x, seq, U_target):
    U = candidate_unitary(seq, x)
    theta_opt = optimal_global_phase(U, U_target)
    U_adjusted = np.exp(-1j * theta_opt) * U
    return np.linalg.norm(U_adjusted - U_target, ord='fro')

def all_candidate_sequences(distinct_couplings):
    left_side = distinct_couplings[:-1]
    center = distinct_couplings[-1]
    candidates = []
    for perm in itertools.permutations(left_side):
        candidate_seq = list(perm) + [center] + list(reversed(perm))
        candidates.append(candidate_seq)
    return candidates

def optimize_candidate(candidate_seq, U_target, k, num_trials, threshold):
    best_cost = np.inf
    best_params_candidate = None
    for trial in range(num_trials):
        x0 = []
        for i in range(k):
            x0.append(np.random.uniform(0, 2))
            x0.append(np.random.uniform(0, 1))
        x0 = np.array(x0)
        bounds = []
        for i in range(k):
            bounds.append((0, 1.999999))
            bounds.append((0, 1))
        res = minimize(cost_function, x0, args=(candidate_seq, U_target),
                       method='L-BFGS-B', bounds=bounds,
                       options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000})
        if res.success and res.fun < best_cost:
            best_cost = res.fun
            best_params_candidate = res.x
            if best_cost < threshold:
                break
    # Print the candidate sequence and its cost
    print("Candidate sequence:", candidate_seq, "Cost: {:.2e}".format(best_cost))
    return candidate_seq, best_params_candidate, best_cost

###########################################
# Code inside the main guard for multiprocessing
###########################################

if __name__ == '__main__':
    # Recompute U_target:
    couplings_orig = ([(2,0),(0,1),(2,0)] +
                      [(4,0),(3,0)]  +
                      [(0,1),(3,0)] +
                      [(0,2),(4,0)] +
                      [(6,0),(0,5),(6,0)] +
                      [(8,0),(7,0)]  +
                      [(0,5),(7,0)] +
                      [(6,0)] +
                      [(0,2),(6,0)] +
                      [(5,0),(0,1),(5,0)] +
                      [(7,0),(0,3),(7,0)] +
                      [(0,4),(8,0)])
    
    fractions_orig = ([1, 1/2, 1] +
                      [1, 1/2] +
                      [1/2, 1] +
                      [1/2, 1] +
                      [1, 1/2, 1] +
                      [1, 1/2] +
                      [1/2, 1] +
                      [1/2] +
                      [1/2, 1] +
                      [1, 1/2, 1] +
                      [1, 1/2, 1] +
                      [1/2, 1])
    
    fixed_phase_flags_orig = ([0.5, 0.5, 0.5] +
                              [0.5, 0.5] +
                              [0.5, 0.5] +
                              [0.5, 0.5] +
                              [0.5, 0.5, 0.5] +
                              [0.5, 0.5] +
                              [0.5, 0.5] +
                              [0.5] +
                              [0.5, 0.5] +
                              [0.5, 0.5, 0.5] +
                              [0.5, 0.5, 0.5] +
                              [0.5, 0.5])
    
    pulses_full = []
    for cpl, frac, ph in zip(couplings_orig, fractions_orig, fixed_phase_flags_orig):
        cpl_fixed, ph_fixed = canonicalize(cpl, ph)
        pulses_full.append((cpl_fixed, frac, ph_fixed))
    
    U_target = np.eye(dim, dtype=complex)
    for (cpl, frac, ph) in pulses_full:
        U_target = pulse_operator(cpl, frac, ph) @ U_target
    U_target = U_target / np.linalg.det(U_target)
    
    print("Target Unitary (rounded):")
    print(np.round(U_target, 3))
    
    distinct_couplings = [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8)]
    all_candidates = all_candidate_sequences(distinct_couplings)
    print("\nTotal number of candidate sequences:", len(all_candidates))
    
    k = 15
    threshold = 1e-6
    num_trials = 10
    
    best_seq_overall = None
    best_params_overall = None
    best_cost_overall = np.inf
    
    start_time = time.time()
    
    with ProcessPoolExecutor() as executor:
        print('starting')
        futures = {executor.submit(optimize_candidate, cand, U_target, k, num_trials, threshold): cand for cand in all_candidates}
        print('defined')
        for i, future in enumerate(as_completed(futures), 1):
            candidate_seq, params, cost_val = future.result()
            print(candidate_seq)
            print(cost_val)
            if cost_val < best_cost_overall:
                best_cost_overall = cost_val
                best_seq_overall = candidate_seq
                best_params_overall = params
                if best_cost_overall < threshold:
                    break
            if i % 100 == 0:
                print(f"Processed {i} candidates; current best cost: {best_cost_overall:.2e}")
    
    end_time = time.time()
    print(f"\nGlobal search completed in {end_time - start_time:.1f} seconds.")
    
    if best_seq_overall is not None:
        print("\nOptimized candidate sequence (couplings):")
        print(best_seq_overall)
        print("\nOptimized parameters (per pulse):")
        for i, coupling in enumerate(best_seq_overall):
            f_val = best_params_overall[2*i]
            phi_val = 0 if best_params_overall[2*i+1] < 0.5 else 1
            print(f"Pulse {i+1}: coupling = {coupling}, fraction = {f_val:.4f}, phase = {phi_val}")
        
        U_optimized = candidate_unitary(best_seq_overall, best_params_overall)
        theta_opt = optimal_global_phase(U_optimized, U_target)
        U_adjusted = np.exp(-1j*theta_opt) * U_optimized
        print("\nTarget Unitary (rounded):")
        print(np.round(U_target, 3))
        print("\nOptimized Candidate Sequence Unitary (rounded):")
        print(np.round(U_adjusted, 3))
        print(f"\nBest cost: {best_cost_overall:.2e}")
    else:
        print("No candidate sequence was found that approximates the target unitary within the threshold.")
