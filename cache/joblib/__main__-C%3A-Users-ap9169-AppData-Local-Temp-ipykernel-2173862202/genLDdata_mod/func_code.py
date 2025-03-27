# first line: 1
@memory.cache
def genLDdata_mod(seed: int | None = 1):
  if seed is not None:
    np.random.seed(seed)
  
  # Sample from the surface of an ellipsoid
  X1, X2, X3 = np.random.randn(3, 1000)
  lambda_ = np.sqrt((X1/4)**2 + (X2/5)**2 + (X3/6)**2)
  X1, X2, X3 = X1 / lambda_, X2 / lambda_, X3 / lambda_
  X = np.column_stack((X1, X2, X3))
  
  # Sample from a cube
  X1, X2, X3 = np.random.rand(3, 1000) + 2
  XX = np.column_stack((X1, X2, X3))
  
  # Sample of the curve
  t = np.random.uniform(-2*np.pi, 2*np.pi, 1000)
  X1 = (t * np.cos(t))/(1 + t**2)
  X2 = (t * np.sin(t))/(1 + t**2)
  X3 = t
  X_curve = np.column_stack((X1, X2, X3))
  
  # Sample from segment connecting ellipsoid and the cube
  point1 = X[np.random.randint(len(X))]
  point2 = XX[np.random.randint(len(XX))]
  t = np.linspace(0, 1, 1000)
  # linear interpolation
  X1 = point1[0] + t * (point2[0] - point1[0])
  X2 = point1[1] + t * (point2[1] - point1[1])
  X3 = point1[2] + t * (point2[2] - point1[2])
  X_segment = np.column_stack((X1, X2, X3))
  
  A = np.vstack((X, XX, X_curve, X_segment))
  return A
