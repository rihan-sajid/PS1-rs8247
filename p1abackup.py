import numpy as np
from scipy.special import exp1
import time
import matplotlib.pyplot as plt

def direct_solve(N, h, rho, xi, yj):
    V = 0.0
    # Sum strictly from 0 to N-1 as per Equation (5)
    for i_prime in range(N):
        for j_prime in range(N):
            # Calculate relative distance
            dx = xi - (i_prime * h)
            dy = yj - (j_prime * h)
            
            # Use the regularized Kernel formula (Equation 4)
            G = -1/(4*np.pi) * np.log(dx**2 + dy**2 + h**2)
            V += rho[i_prime, j_prime] * G
    return V

def get_exact_V(x, y, xc, yc, sigma, h):
    # r^2 must be regularized with + h^2 to match the numerical kernel
    r2 = (x - xc)**2 + (y - yc)**2 + h**2 # Regularization added here
    z = r2 / (2 * sigma**2)
    # Equation (7)
    V_exact = -1.0 / (4.0 * np.pi) * (exp1(z) + np.log(r2))
    return V_exact

def main():
    N = [8, 16, 32, 64]
    times = []
    L_inf = []
    sigma = 0.05
    x_c = 0.5
    y_c = 0.5

    for n in N:
        print(f"Running for N={n}...")
        h = 1.0 / n 
        rho = np.zeros((n+1, n+1))
        V_ij = np.zeros((n+1, n+1))
        V_sol = np.zeros((n+1, n+1))

        # 1. Fill the charge distribution
        for i in range(n+1):
            for j in range(n+1):
                rho[i, j] = 1/(2*np.pi*sigma**2)*np.exp(-1*((i*h-x_c)**2+(j*h-y_c)**2)/(2*sigma**2))

        # 2. Time ONLY the solver 
        start_time = time.perf_counter()
        for i in range(n+1):
            for j in range(n+1):
                V_ij[i, j] = h**2 * direct_solve(n, h, rho, i*h, j*h)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        # 3. Calculate exact solution separately
        for i in range(n+1):
            for j in range(n+1):
                V_sol[i, j] = get_exact_V(i*h, j*h, x_c, y_c, sigma, h)

        error = np.abs(V_ij - V_sol)
        # use nan-aware max: the exact solution is singular at the center (NaN)
        L_inf.append(np.nanmax(error))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.loglog(N, times, marker='o')
    plt.xlabel('N')
    plt.ylabel('Time (seconds)')    
    plt.title('Time taken for direct solve vs N')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.loglog(N, L_inf, marker='o')
    plt.xlabel('N') 
    plt.ylabel('L_inf Error')
    plt.title('L_inf Error vs N')
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()