import numpy as np
import matplotlib.pyplot as plt
import time

def get_phi(x, y):
    """Analytical function defined in Eq (10)."""
    # phi(x,y) = sin(4*pi*x^2) * cos(2*pi*y^3)
    return np.sin(4 * np.pi * x**2) * np.cos(2 * np.pi * y**3)

def get_true_gradient(x, y):
    """Analytical gradient defined in Eq (11)."""
    # Using the components provided in the PDF
    df_dx = 8 * np.pi * x * np.cos(4 * np.pi * x**2) * np.cos(2 * np.pi * y**3)
    df_dy = -6 * np.pi * y**2 * np.sin(4 * np.pi * x**2) * np.sin(2 * np.pi * y**3)
    return df_dx, df_dy

def gradient_solve_4th_order(N, h):
    """
    Computes the gradient of phi using 4th-order central differences.
    Uses padding to maintain accuracy at boundaries.
    """
    # 1. Create a padded grid (add 2 ghost cells on every side)
    # Range is from -2h to (1 + 2h)
    pad = 2
    idx_padded = np.arange(-pad, N + pad + 1)
    x_pad = idx_padded * h
    y_pad = idx_padded * h
    X_pad, Y_pad = np.meshgrid(x_pad, y_pad, indexing='ij')
    
    # 2. Evaluate phi on the padded grid
    f_padded = get_phi(X_pad, Y_pad)
    
    # 3. Apply 4th-order stencil: (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / 12h 
    # We slice the padded array so the output is exactly (N+1, N+1)
    # For grad_x: we shift indices in the 'i' (row) direction
    grad_x = (
        -f_padded[4:, 2:-2] + 8*f_padded[3:-1, 2:-2] - 
         8*f_padded[1:-3, 2:-2] + f_padded[:-4, 2:-2]
    ) / (12 * h)
    
    # For grad_y: we shift indices in the 'j' (column) direction
    grad_y = (
        -f_padded[2:-2, 4:] + 8*f_padded[2:-2, 3:-1] - 
         8*f_padded[2:-2, 1:-3] + f_padded[2:-2, :-4]
    ) / (12 * h)
    
    return grad_x, grad_y

def main():
    # Use powers of 2 for N to see clear convergence [cite: 55]
    Ns = [16, 32, 64, 128, 256]
    errors = []
    hs = []

    for n in Ns:
        h = 1.0 / n
        hs.append(h)
        
        # Numerical Gradient
        grad_x_num, grad_y_num = gradient_solve_4th_order(n, h)
        
        # Analytical Gradient on the (N+1, N+1) grid 
        x = np.linspace(0, 1, n + 1)
        y = np.linspace(0, 1, n + 1)
        X, Y = np.meshgrid(x, y, indexing='ij')
        grad_x_true, grad_y_true = get_true_gradient(X, Y)
        
        # L2 Error calculation as per Eq (12) [cite: 74]
        # ||e|| = sqrt( h^2 * sum( |grad_true - grad_num|^2 ) )
        diff_sq = (grad_x_true - grad_x_num)**2 + (grad_y_true - grad_y_num)**2
        l2_error = np.sqrt(h**2 * np.sum(diff_sq))
        errors.append(l2_error)
        
        print(f"N={n:3d} | h={h:.5f} | L2 Error={l2_error:.2e}")

    # Plotting [cite: 78, 79]
    plt.figure(figsize=(8, 6))
    plt.loglog(hs, errors, '-o', label='Numerical Error')
        
    plt.xlabel('Grid Spacing (h)')
    plt.ylabel('L2 Norm of Error')
    plt.title('Convergence Study: 4th-Order Central Difference')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig('p2b.png')
    plt.show()

if __name__ == "__main__":
    main()