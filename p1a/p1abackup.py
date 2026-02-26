import matplotlib
import numpy as np
from scipy.special import exp1
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')   

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

def fft_solve(N, h, rho):
    
    rho_adjusted = np.zeros((2*N, 2*N))
    rho_adjusted[:N, :N] = rho[:N, :N]  
    G = np.zeros((2*N, 2*N))

    i = np.arange(2*N)
    i_dist = np.where(i <= N, i, i - 2*N) 
    X_dist, Y_dist = np.meshgrid(i_dist * h, i_dist * h, indexing='ij')
    
    G = -1/(4*np.pi) * np.log(X_dist**2 + Y_dist**2 + h**2)

    # Inverse FFT to get potential in real space
    G_inv = np.fft.fft2(G)
    rho_fft = np.fft.fft2(rho_adjusted)

    V_fft = G_inv * rho_fft
    V = (h**2)*np.fft.ifft2(V_fft).real
        
    return V

def get_exact_V(x, y, xc, yc, sigma, h):
    # r^2 must be regularized with + h^2 to match the numerical kernel
    r2 = (x - xc)**2 + (y - yc)**2 + h**2 # Regularization added here
    z = r2 / (2 * sigma**2)
    # Equation (7)
    V_exact = -1.0 / (4.0 * np.pi) * (exp1(z) + np.log(r2))
    return V_exact

def main():
    N = [8, 16, 32, 64, 128]
    times_rectangle = []
    times_fft = []
    L_inf_rectangle = []
    L_inf_fft = []
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
        times_rectangle.append(end_time - start_time)

        start_time = time.perf_counter()
        V_fft = fft_solve(n, h, rho)
        end_time = time.perf_counter()
        times_fft.append(end_time - start_time)

        # 3. Calculate exact solution separately
        for i in range(n+1):
            for j in range(n+1):
                V_sol[i, j] = get_exact_V(i*h, j*h, x_c, y_c, sigma, h)

        error = np.abs(V_ij - V_sol)
        # use nan-aware max: the exact solution is singular at the center (NaN)
        L_inf_rectangle.append(np.nanmax(error))

        error_fft = np.abs(V_fft[:n+1, :n+1] - V_sol)
        L_inf_fft.append(np.nanmax(error_fft))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.loglog(N, times_rectangle, marker='o', label='Direct Solve')
    plt.loglog(N, times_fft, marker='s', label='FFT Solve')
    plt.xlabel('N')
    plt.ylabel('Time (seconds)')    
    plt.title('Time taken for direct solve vs N')
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.loglog(N, L_inf_rectangle, marker='o', label='Direct Solve')
    plt.loglog(N, L_inf_fft, marker='s', label='FFT Solve') 
    plt.xlabel('N') 
    plt.ylabel('L_inf Error')
    plt.title('L_inf Error vs N')
    plt.grid()
    plt.legend()
    plt.savefig('p1c.png')

if __name__ == "__main__":
    main()