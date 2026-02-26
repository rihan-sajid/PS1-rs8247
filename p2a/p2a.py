import matplotlib
import numpy as np
from scipy.special import exp1
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')   

def gradient_solve(f, h):
    # Compute the gradient using central differences
    grad_x = np.zeros_like(f)
    grad_y = np.zeros_like(f)
    
    # Central difference for interior points
    grad_x[1:-1, :] = (-f[:-2, :] + 8*f[1:-1, :] - 8*f[2:, :] + f[3:, :]) / (12 * h)
    grad_y[:, 1:-1] = (-f[:, :-2] + 8*f[:, 1:-1] - 8*f[:, 2:] + f[:, 3:]) / (12 * h)
    
    # Forward/backward difference for boundaries
    grad_x[0, :] = (f[1, :] - f[0, :]) / h
    grad_x[-1, :] = (f[-1, :] - f[-2, :]) / h
    grad_y[:, 0] = (f[:, 1] - f[:, 0]) / h
    grad_y[:, -1] = (f[:, -1] - f[:, -2]) / h
    
    return grad_x, grad_y

def main():
    N = [8, 16, 32, 64, 128]
    
    for n in N:
        h = 1.0 / n
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        f = np.sin(4*np.pi * X**2) * np.cos(2*np.pi * Y**3)
        
        start_time = time.time()    
        grad_x, grad_y = gradient_solve(f, h)
        end_time = time.time()
        
        print(f"N={n}, Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":    
    main()