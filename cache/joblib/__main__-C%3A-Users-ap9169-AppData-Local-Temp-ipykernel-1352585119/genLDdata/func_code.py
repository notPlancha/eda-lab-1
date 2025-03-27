# first line: 1
@memory.cache
def genLDdata(seed: int | None = 1):
  if seed is not None:
    np.random.seed(seed)
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
  return A
