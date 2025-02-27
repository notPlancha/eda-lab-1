# generated from genLDdata.m: https://chatgpt.com/share/67c0c504-f004-8001-80b5-b8746ae6270a
import numpy as np
import matplotlib.pyplot as plt

def genLDdata(plot=True):
    # Sample from the surface of a sphere
    X1, X2, X3 = np.random.randn(3, 1000)
    lambda_ = np.sqrt(X1**2 + X2**2 + X3**2)
    X1, X2, X3 = X1 / lambda_, X2 / lambda_, X3 / lambda_
    X = np.column_stack((X1, X2, X3))
    
    # Sample from a cube
    X1, X2, X3 = np.random.rand(3, 1000) + 2
    XX = np.column_stack((X1, X2, X3))
    
    # Sample from lines attached to a sphere
    L1 = np.column_stack((np.zeros(1000), np.zeros(1000), 2 * np.random.rand(1000) + 1))
    L2 = np.column_stack((np.zeros(1000), np.zeros(1000), -2 * np.random.rand(1000) - 1))
    L3 = np.column_stack((np.zeros(1000), 2 * np.random.rand(1000) + 1, np.zeros(1000)))
    L4 = np.column_stack((np.zeros(1000), -2 * np.random.rand(1000) - 1, np.zeros(1000)))
    
    A = np.vstack((X, XX, L1, L2, L3, L4))
    
    if plot:
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='r', marker='.')
        ax.scatter(XX[:, 0], XX[:, 1], XX[:, 2], c='g', marker='.')
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='b', marker='.')
        ax.grid(True)
        plt.show()
    
    return A

# Call the function to generate data and plot
if __name__ == "__main__":
  genLDdata()
