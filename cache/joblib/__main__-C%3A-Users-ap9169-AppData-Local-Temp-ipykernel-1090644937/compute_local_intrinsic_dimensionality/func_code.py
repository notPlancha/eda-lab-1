# first line: 1
@memory.cache
def compute_local_intrinsic_dimensionality(A, method, k=100):
  # Compute pairwise Euclidean distances
  Ad = dist.squareform(dist.pdist(A))
  
  # Get the dimensions of A
  nr, nc = A.shape
  Ldim = np.zeros(nr)
  
  # Sort distances and get indices
  Ads = np.sort(Ad, axis=1)
  J = np.argsort(Ad, axis=1)
  
  # Compute local intrinsic dimensionality
  for m in range(nr):
    Ldim[m] = method(A[J[m, :k], :])
  
  # Adjust local dimensions
  Ldim[Ldim > 3] = 4
  Ldim = np.ceil(Ldim).astype(int)
  
  # Tabulate results
  unique, counts = np.unique(Ldim, return_counts=True)
  percentages = (counts / nr) * 100
  tabulation_df = pd.DataFrame({'Dimension': unique, 'Count': counts, 'Percentage': np.round(percentages, 3)})
  
  return Ldim, tabulation_df
