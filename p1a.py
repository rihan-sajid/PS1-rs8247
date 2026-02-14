import numpy as np
from scipy.special import exp1
import time
import matplotlib.pyplot as plt

def direct_solve(N, h, rho, x, y):
    V = 0.0
    G = 0.0
    for i in range(N+1):
        for j in range(N+1):
            G = -1/(4*np.pi)*np.log((x-i*h)**2+(y-j*h)**2 + h**2)
            V = V + rho[i, j]*G
    return V

def get_exact_V(x, y, h, xc, yc, sigma):
    # Calculate the squared distance from the center (xc, yc)
    r2 = (x - xc)**2 + (y - yc)**2
    z = r2 / (2 * sigma**2)
    V_exact = 1 / (4 * np.pi) * (exp1(z) + np.log(r2))

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
        rho = np.zeros((n + 1, n + 1))
        V_ij = np.zeros((n + 1, n + 1))
        V_sol = np.zeros((n + 1, n + 1))

        # 1. Fill the charge distribution
        for i in range(n + 1):
            for j in range(n + 1):
                rho[i, j] = 1/(2*np.pi*sigma**2)*np.exp(-1*((i*h-x_c)**2+(j*h-y_c)**2)/(2*sigma**2))

        # 2. Time ONLY the solver 
        start_time = time.perf_counter()
        for i in range(n + 1):
            for j in range(n + 1):
                V_ij[i, j] = h**2 * direct_solve(n, h, rho, i*h, j*h)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

        # 3. Calculate exact solution separately
        for i in range(n + 1):
            for j in range(n + 1):
                V_sol[i, j] = get_exact_V(i*h, j*h, h, x_c, y_c, sigma)

        error = np.abs(V_ij - V_sol)
        L_inf.append(np.max(error))

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