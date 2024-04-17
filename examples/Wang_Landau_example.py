import numpy as np
import matplotlib.pyplot as plt

def wang_landau(E_max, flatness_criteria=0.8, final_ln_f=1e-8):
    E_range = np.arange(E_max + 1)  # Energy levels from 0 to E_max
    g = np.zeros(E_max + 1)  # ln(g(E)), logarithm of the density of states
    H = np.zeros(E_max + 1)  # Histogram of energy visits

    E = np.random.randint(0, E_max+1)  # Current energy state, randomly initialized
    ln_f = 1.0  # Initial modification factor (ln(f))
    iteration = 0  # Counter for iterations

    while ln_f > final_ln_f:
        iteration += 1
        # Propose a random transition to a neighboring energy level
        proposed_E = E + np.random.choice([-1, 1])
        # Apply periodic boundary conditions
        if proposed_E < 0 or proposed_E > E_max:
            proposed_E = E  # Reject moves that go out of the allowed energy range
        
        # Metropolis-Hastings acceptance criterion (in log form)
        if np.log(np.random.rand()) < g[E] - g[proposed_E]:
            E = proposed_E  # Accept move

        # Update the density of states and the histogram
        g[E] += ln_f
        H[E] += 1

        # Check for histogram "flatness"
        if iteration % 10000 == 0:  # Check flatness every 10000 iterations
            norm_H = H / np.max(H)
            if np.min(norm_H[norm_H > 0]) > flatness_criteria:  # Flatness condition
                print(f"Reducing ln_f: {ln_f} -> {ln_f / 2} at iteration {iteration}")
                ln_f /= 2  # Reduce the modification factor
                H[:] = 0  # Reset histogram

    return E_range, np.exp(g - np.max(g))  # Return normalized g(E)

# Parameters
E_max = 10  # Maximum energy level

# Run the Wang-Landau algorithm
E_range, g_E = wang_landau(E_max)

# Plotting the density of states
plt.figure(figsize=(10, 6))
plt.bar(E_range, g_E, color='blue', alpha=0.7)
plt.title('Density of States estimated by Wang-Landau Algorithm')
plt.xlabel('Energy')
plt.ylabel('Density of States (normalized)')
plt.grid(True)
plt.show()
