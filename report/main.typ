#set text(font: "Stix Two Text", size: 16pt, weight: "medium")
#set par(spacing: 1em)
#import "@preview/codly:1.2.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(languages: codly-languages, zebra-fill: none)
#show figure.where(
  kind: table
): set figure.caption(position: top)

#let today = datetime.today()
#align(center)[EDA Lab 1]
#set text( size: 14pt)
#grid(columns: (1fr, 1fr), [
    André Plancha \ #text(size:7pt)[#link("mailto:andre.plancha@hotmail.com")] \
    Yana Zlatanova \ #text(size:7pt)[#link("mailto:yana-zlatanova-gold@gmail.com")]
  ],align(right)[
      MALMÖ UNIVERTITY \
      #align(horizon)[MA661E - VT25]
      #align(bottom)[#today.display("[month repr:long] [day], [year]")]
]
)
#v(-6pt)
#line(length: 100%)
#set text(size: 12pt, weight: "regular")
#set par(justify: true, spacing: 1.2em)
#set table(stroke: none)
// #show table: set block(fill: none, width: auto)
#let question(it) = {
  show heading: itt => text(size: 12pt)[#itt.body]
  block(fill: luma(220), inset: 8pt, width: 100%, radius: 1pt, it)
}

#set text(font: "Stix Two Text", size: 16pt, weight: "medium")

#set text(size: 12pt, weight: "regular")
#set par(justify: true)
#set table(stroke: none)

#let question(it) = {
  show heading: itt => text(size: 12pt)[#itt.body]
  block(fill: luma(220), inset: 8pt, width: 100%, radius: 1pt, it)
}

#show table: set align(center)

#question[
  = 1) 
  Generate n =150, p = 6 normally distributed random variables that have high variance in some dimension and low variance in another dimension. Try PCA using both correlation and covariance matrices.
]
Following the example given, a dataset with 2 dimensions of high variance and a dataset with 4 dimensions of low variance were created and combined into one dataset with $n = 150, p = 6$.

#question[
  == a) Is the covariance matrix very informative?  
]
The covariance matrix PCA is not very informative in our example because it reflects only the variance of the the first 2 dimensions with high-variance and it fails to show the structures in the remaining dimensions. 

#question[
  == b) *Which one would be better to use in this case?*
]
 In our case correlation matrix would be a better choice because it standardizes all variables to have equal variance before finding principal components.


#figure(
  image("images/PCA_covariance.png", width: 70%), caption: [PCA using covariance matrix ]
)

#figure(
  image("images/PCA_CORR.png", width: 70%), caption: [PCA using correlation matrix ]
)


#question[ 
  = 2)
  For each case, apply Linear Discriminant Analysis (LDA) to the `Iris` dataset and visualize the data in one dimension, using the Kernel Density Estimate; and discuss how good the mapping are for each case.

  == case a):
  Apply LDA for only 2 classes at a time, i.e., [setosa, versicolor]; [versicolor, virginica]; [virginica, setosa].
]
The `Iris` dataset is composed of 150 samples of iris flowers with 4 numerical features that describe sepal length, sepal width, petal length and petal width. Each flower can be a setosa, a versicolor, or a virginica (target).

For this case, for each pair, we filtered out the other class and applied `sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)` to the remaining rows, using the class as the target. To calculate and plot Kernel Density Estimate, `plotnine.geom_density` was used, which by default uses a _Gaussian_ kernel. The code for this process can be observed in @2a.

#figure(
  image("images/2a_set_ver.png"), caption: [LDA without virginica]
) <2a_set_ver>
As observed in @2a_set_ver, since there's no overlap between the mappings, the LDA transformation from the _setosa_ and _versicolor_ pair effectively found a direction where the two classes are linearly separable.
#figure(
  image("images/2a_set_vir.png"), caption: [LDA without versicolor]
)
Similar to the previous pair, the _setosa_ and _virginica_ pair also shows no overlap between the mappings, so the LDA transformation effectively found a direction where the two classes are linearly separable as well.


#figure(
  image("images/2a_ver_vir.png"), caption: [LDA without setosa]
)<2a_ver_vir>

However, since there's some overlap in the transformation from the _versicolor_ and _virginica_ pair, as observed in @2a_ver_vir we can conclude that the LDA couldn't find a direction as effective as the other ones; with that being said, the overlap is quite small, so it can be argued that it's still a good mapping.


#question[
  == case b):
  Instead of classifying based on species, consider two broad classes—Sepals and Petals. Apply LDA to distinguish between sepal-based features ([sepal length, sepal width]) and petal-based features ([petal length, petal width]).
]
For this case, we decided on concatenating the sepal width/length and petal width/length to make a `Width`/`Lenght` column, and make the target `Sepal_or_Petal` describe if the row was produced from the Sepal or Petal part. A sample of this transformed dataset can be seen on @2b_t.

#figure(image("SP_table.png", height: 20%), caption: [Sepal or Petal dataset sample])<2b_t>

After this, a similar process from *2) a)* was done to for the LDA transformation and Kernel Density Estimation. The code for this one can be observed in @2b_code.

#figure(
  image("images/2b.png"),
)<2b>

As observed in @2b, there's some small overlap between the mappings, which suggests that the direction found by the LDA is somewhat effective. 

#question[
  = 3) 
  *Factory analysis*
  
  == a) 
  *Repeat example 2.5 for dataset stockreturns.*
]

For this analysis, we worked with the stockreturns dataset containing 10 columns (companies) and 100 rows (trading days, assumed). We first standardized the data and then applied factor analysis to extract three factors. We implemented the analysis both with varimax rotation and without rotation for comparison.

The 2D factor loading plots below visualize how each company relates to the first two factors, revealing which companies tend to move together and which respond differently to the factors.

#figure(
  image("task3-2D-loading-plot-rotation.png", width: 90%), caption: [ Factor loading plot with varimax rotation ]
)

#figure(
  image("task3-2D-loading-plot-NO-rotation.png", width: 90%), caption: [ Factor loading plot with no rotation ]
)

If we compare the two figures, it's evident that rotation creates a more differentiated structure. The un-rotated solution primarily separates companies along Factor 2, while the rotated solution identifies more distinct groupings that better reflect underlying relationships.

The figure below shows a few plots that display the factor scores from the factor analysis with rotation, where each point represents a single trading day positioned according to its scores on each factor. These plots reveal varied market behavior across trading days with no distinct clustering patterns. 

#figure(
  image("task3-FA-rotation-3-factor-compared.png", width: 80%), caption: [ Factor scores from the factor analysis with rotation ]
)


#question[
  == b) 
  Carry out a factor analysis for your data for 10 companies over 20 days.
]

Based on Figure 11, SEB and Nordea bank are positioned close together, indicating they have similar factor loadings and patterns of correlation with the underlying factors. 

#figure(
  image("task3b-3D-text-lables.png", width: 90%), caption: [ Factor loading plot with rotation ]
)

Figure 12 illustrates a broad distribution of points across all three plots, confirming that each component captures variation in the data. However, the absence of clear linear patterns between any pair of components suggests that the components are relatively independent.

#figure(
  image("task3b-components.png"), caption: [ Factor scores from the factor analysis with rotation ]
)



#question[
  = 4)
  Using the following curve:

  $x_1 = (t cos t)/(1 + t^2), x_2 = (t sin t)/(1 + t^2), x_3 = t, -2pi <= t <= 2pi$,
  
  == a) 
  Estimate the intrinsic dimensionality using the Pettis, Bailey, Jain, and Dubes algorithm (available in `idpettis.m`)
]

The curve was generated using the code listed in @4a_code and the curve can be visualized in @4a.

#figure(image("images/4a).png", height: 30%), caption: [Curve generated]) <4a>

To calculate the intrinsic dimensionality estimate, `pyEDAkit.IntrinsicDimensionality.id_pettis` was used, resulting in 
$ "IDE" approx 1.119 $

#question[
  == b)
  Study the intrinsic dimensionality when introducing noise of various sizes to the curve.
]

For this exercise, to introduce noise of various sizes, we decided to generate and add random points to our plot, and record the intrinsic dimensionality every time a new random point was added. The random points were uniformly generated over the minimum and maximum of every dimension of our curve. The code for this process is listed in @4b_code, the curve with the final noise can be observed in @4b_figure, and the Estimate Intrinsic Dimensionality plotted over the number of noise data points added can be observed in @4b_plot

#figure(
  image("images/4b_figure.png", height: 20%), caption: [Curve with noise added]
)<4b_figure>

#figure(
  image("images/4b plot.png", height: 40%), caption: [Intrinsic Dimensionality over noise size]
)<4b_plot>

For the first $approx 12$ noise data points, the intrinsic Dimensionality seems to be relatively low and going down, suggesting that the dataset has a we--defined structure; until it reaches a somewhat consistent value until $approx 27$, where the values starts increasing rapidly with the number of noise data points added. This suggest that beyond a certain point, as noise increases, the metric also increases, showing how noise disrupts rapidly disrupts the original structure.

#question[
  == c)
  Is there any threshold number of noise size for the intrinsic dimensionality estimate?
]

While it's difficult to define a strict threshold, for this particular curve, a noise level of approximately 40 data points can be considered a reasonable threshold.

#question[
  = 5)
  
  == a) 
  Apply the Singular value decomposition (SVD) to dataset Leukemia, choose a proper lower dim $k$ via elbow in the plot of singular vaules, then plot the dimension reduced data in both 2-dim and 3-dim (in case your $k$ is at least three).
]
With the elbow method we identified $k = 6$ as the optimal number of components for dimensionality reduction, as shown on Figure 16. Using this information, we then performed Singular Value Decomposition (SVD) and visualized the reduced-dimension data in both 2D and 3D, as shown in Figure 17 and 18.

#figure(image("task5-elbow.png", width: 60%), caption: [Elbow method.])

#figure(image("task5a-2d-SVD.png", width: 70%), caption: [Singular Value Decomposition 2D visualization.])

#figure(image("task5b-3d-SVD.png", width: 70%), caption: [Singular Value Decomposition 3D visualization.])

#question[
  == b) 
  *Try PCA (covaraince) and compare the results from these two different methods.*
]
From the figures below the results of Singular Value Decomposition (SVD) and Principal Component Analysis (PCA) applied to the same leukemia dataset can be compared. Both methods reveal similar distribution patterns with a significant difference in the range of the second component, where for SVD it spans from -13 to 3, and for PCA from -3 to 13. Additionally, the SVD results show a higher concentration of points in the upper half of the plot, while PCA points clustered mostly in the lower half. However, the overall shape and relative positioning of the points somehow appears consistent, indicating that both methods capture similar underlying patterns in the data.

  #figure(image("task5b-PCA-2D.png", width: 60%), caption: [Singular Value Decomposition 3D visualization.])
  #figure(image("task5a-2d-SVD.png", width: 60%), caption: [Singular Value Decomposition 2D visualization.])



#question[
  = 6) 
  == a)
  Using `genLDdata.m`, calculate the global and local intrinsic dimensionalities, using `MLE` and `CorrDim`, and compare the results. Use a neighborhood size of $k = 100$.
]
#let gid = [global intrinsic dimensionality]
#let lid = [local intrinsic dimensionality]
`genLDdata` generates random data points from the surface of a sphere, from a cube, and from 4 horizontal or vertical line segments coming out of the sphere. The #gid using MLE of the generated data was $approx 1.52$, and using _CorrDim_ was $approx 1.00$. The #lid counts are presented in @table_lid, and a scatter plot of these are available on @sp_mle and @sp_corrdim.

#figure(table(columns: 3,
  table.header([*Dimension*],[*Count (MLE)*], [*Count (CorrDim)*]),
  table.hline(),
  [1], [2798 (46.63%)], [2749 (45.82%)],
  [2], [1922 (32.03%)], [2170 (36.17%)],
  [3], [1257 (20.95%)], [1080 (18.00%)],
  [4], [23 (0.38%)], [1 (0.02%)],
), caption: [Local Intrinsic Dimensionality Counts])<table_lid>

#figure(image("images/6a_w_mle.png"), caption: [Local Intrinsic Dimensionality Scatter plot with `MLE`])<sp_mle>
#figure(image("images/6A_W_CORR.png"), caption: [Local Intrinsic Dimensionality Scatter plot with `CorrDim`])<sp_corrdim>

From the figures, we can see that each shape, excluding the sphere surface, has a distinct #lid that matches our geometric intuition: the cube completely displays a #lid of 3 with both algorithms and the segments mostly displays a #lid of 1; however, since the sphere surface is a more complex surface, it seems the different algorithms classify it differently and not consistently: the output from the _MLE_ presents the #lid of the sphere to be between 2 and 3, while the one with _CorrDim_ presents it to be between 1 and 2 (the 1 points are closer to the connection between the segments and the sphere). Lastly, it seems the _MLE_ could classify the segments more consistently than _CorrDim_.

#question[
  == b)
  Modify `genLDdata.m` so that:
  - Instead of the sphere, we have the surface of an ellipsoid of $x^2_1/4^2 + x^2_2/5^2 + x^2_3/6^2 = 1$,
  - The line segments are replaced by the curve $x_1 = (t cos t^2)/(1+t^2), x_2 = (t sin t^2)/(1+t^2), x_3 = t, -2pi <= t <= 2pi$
  - A line segment connecting a point from the ellipsoid and a point from the cube is added
  Using this modified method, study the global and local intrinsic dimensionalities using `PackingNumbers`
]

The modified function can be found on @genLDdata_mod and its output can be visualized on @sp_pn, along with the output of the #lid with _PackingNumbers_. The #gid was $approx$ #calc.round(0.6918028005414134, digits:2)

#figure(
image("images/Local Intrinsic Dimensionality with _PackingNumbers_.png"), caption: [Local Intrinsic Dimensionality with _PackingNumbers_]
)<sp_pn>

We can see that the algorithm could not effectively find the expected #lid for any of our shapes, besides the connecting line between the cube and the sphere.
// Empirically, this can either be because of the added complexity of the shapes, or because of the different algorithm used


= Appendix
#show figure: set block(breakable: true)
== 1)
#figure(```python
import numpy as np
import matplotlib.pyplot as plt
import pyEDAkit as eda_lin

np.random.seed(42)
x1 = np.random.randn(150, 2) * 100 # 2 dimensions with high variance
x2 = np.random.randn(150, 4) # 4 dimensions with low variance
X = np.hstack((x1, x2))

print("\nPCA using covariance matrix")
PCA_cov = eda_lin.PCA(X, d=3, covariance = True, plot=True)

print("\nPCA using correlation matrix")
PCA_corr = eda_lin.PCA(X, d=2, covariance = False,  plot=True)
```, caption: [PCA using covariance vs correlation matrix for task _1_]) <1_code>

#figure(```python
def PCA(X, d, covariance = True, plot = False):
    X_mean = X.mean(axis=0)
    X = X - X_mean
    S = None
    if covariance:
        S = np.cov(X, rowvar=False)
    else:
        S = np.corrcoef(X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(S)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    Z = (X @ sorted_eigenvectors)[:, :d]

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(sorted_eigenvalue) + 1), sorted_eigenvalue, marker='o', linestyle='-')
        plt.plot(d, sorted_eigenvalue[d - 1], 'ro', label = 'd')
        plt.title('Scree Plot')
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue Magnitude')
        plt.legend()
        plt.grid(True)
        # scatter matrix of Z
        sns.pairplot(pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(d)]), diag_kind='kde')
        plt.legend()
        plt.show()

    return Z
```, caption: [PCA method for task _1_]) <1_code>


== 2)
=== a)
#figure(```python
from plotnine import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

plots = []
lda = LDA(n_components=1)
for target in iris_target_names:
  # filter out target
  iris_filtered = iris[iris["target"] != target]
  Z = lda.fit_transform(iris_filtered.drop(columns="target"), iris_filtered["target"])
  plot = (
    ggplot(
      pd.DataFrame({"Component": Z.flatten(), "target": iris_filtered["target"]}),
      aes(x = 'Component', color = 'target', fill = "target")
    ) 
    + geom_density(alpha = 0.6) 
  )
  plots.append(plot)
plots[0]
plots[1]
plots[2]
```, caption: [LDA for case _a_])<2a>

=== b)

#figure(```python
import numpy as np

data_other_way = pd.DataFrame(
  {
    "Width": np.concatenate([iris["sepal width (cm)"], iris["petal width (cm)"]]),
    "Length": np.concatenate([iris["sepal length (cm)"], iris["petal length (cm)"]]),
    "Sepal_Or_Petal": np.concatenate([np.repeat("Sepal", len(iris)), np.repeat("Petal", len(iris))])
  }
)
Z = lda.fit_transform(data_other_way.drop(columns="Sepal_Or_Petal"), data_other_way["Sepal_Or_Petal"])
(
  ggplot(
    pd.DataFrame({"Component": Z.flatten(), "Sepal_Or_Petal": data_other_way["Sepal_Or_Petal"]}),
    aes(x = 'Component', color = 'Sepal_Or_Petal', fill = "Sepal_Or_Petal")
  ) 
  + geom_density(alpha = 0.6) 
)
```, caption: [LDA for case _b_]) <2b_code>

== 3)
=== a)

#figure(```python
stocks_df = pd.read_excel('../data/stockreturns.xlsx')
stocks = stocks_df.values  # convert to numpy array

scaler = StandardScaler()
stocks_scaled = scaler.fit_transform(stocks)

# FA with rotation (default)
fa_default = FactorAnalysis(n_components=3, rotation='varimax')
fa_default.fit(stocks_scaled)
loadings_default = fa_default.components_.T  

# FA NO rotation
fa_none = FactorAnalysis(n_components=3, rotation=None)
fa_none.fit(stocks_scaled)
loadings_none = fa_none.components_.T
```, caption: [Factor Analysis with and without rotation]) <3a_code2>

#figure(```python
lab = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'] # lables
t = np.linspace(-1, 1, 20) # reference line vector

def plot_factor_loadings(loadings, title):
    plt.figure(figsize=(8, 6))
    plt.plot(loadings[:, 0], loadings[:, 1], '.')
    plt.plot(t, np.zeros_like(t), 'b')  # horizontal line
    plt.plot(np.zeros_like(t), t, 'b')  # vertical line
    
    # add labels
    for i, txt in enumerate(lab):
        plt.annotate(txt, (loadings[i, 0] + 0.02, loadings[i, 1]))
    
    plt.xlabel('Factor 1')
    plt.ylabel('Factor 2')
    plt.title(title, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

plot_factor_loadings(loadings_default, 'Factor Analysis (Varimax)')
plot_factor_loadings(loadings_none, 'Factor Analysis (No rotation)')```, caption: [Plot factor loadings method]) <3a_code3>

#figure(```python
fa_scores = fa_default.transform(stocks_scaled)
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

axes[0].plot(fa_scores[:, 0], fa_scores[:, 1], '*')
axes[0].set_xlabel('Component 1')
axes[0].set_ylabel('Component 2')
axes[0].grid(True)

axes[1].plot(fa_scores[:, 1], fa_scores[:, 2], '*')
axes[1].set_xlabel('Component 2')
axes[1].set_ylabel('Component 3')
axes[1].grid(True)

axes[2].plot(fa_scores[:, 2], fa_scores[:, 0], '*')
axes[2].set_xlabel('Component 3')
axes[2].set_ylabel('Component 1')
axes[2].grid(True)

plt.tight_layout()
plt.show()
```, caption: [Factor scores for Factor Analysis with rotation]) <3a_code4>


=== b)
#figure(```python
data = pd.read_csv('../data/my_stock_returns.csv')

my_stocks_scaled = StandardScaler().fit_transform(data.values)
stock_names = data.columns.tolist()

# FA with varimax rotation
fa = FactorAnalysis(n_components=3, rotation='varimax')
fa.fit(my_stocks_scaled)
loadings = fa.components_.T

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(loadings[:, 0], loadings[:, 1], loadings[:, 2], s=50)
for i in range(len(loadings)):
    ax.text(loadings[i, 0], loadings[i, 1], loadings[i, 2], stock_names[i], size=8)

for spine in ['xy', 'xz', 'yz']:
    ax.grid(True, ls='--', alpha=0.3, which='major', zorder=0)

ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_zlabel('Factor 3')
ax.set_title('3D Factor Analysis loadings plot')

# subplots for component visualization
fa_scores = fa.transform(my_stocks_scaled)
fig, axes = plt.subplots(3, 1, figsize=(8, 12))

axes[0].plot(fa_scores[:, 0], fa_scores[:, 1], '*')
axes[0].set_xlabel('Component 1')
axes[0].set_ylabel('Component 2')
axes[0].grid(True)

axes[1].plot(fa_scores[:, 1], fa_scores[:, 2], '*')
axes[1].set_xlabel('Component 2')
axes[1].set_ylabel('Component 3')
axes[1].grid(True)

axes[2].plot(fa_scores[:, 2], fa_scores[:, 0], '*')
axes[2].set_xlabel('Component 3')
axes[2].set_ylabel('Component 1')
axes[2].grid(True)

plt.tight_layout()
plt.show()
```, caption: [Factor analysis of collected stock data.])<3b_code>

== 4)
=== a)

#figure(```python
import numpy as np
import pandas as pd
import plotly.express as px
np.random.seed(1)

n = 500
theta = np.random.uniform(-2*np.pi, 2*np.pi, n)
x = (theta*np.cos(theta))/(1+theta**2)
y = (theta*np.sin(theta))/(1+theta**2)
z = theta
df = pd.DataFrame({'x':x, 'y':y, 'z':z})

fig = px.scatter_3d(df, x='x', y='y', z='z', color='z')
fig.show()
```, caption: [generating and plotting curve])<4a_code>

#figure(```python
from pyEDAkit.IntrinsicDimensionality import id_pettis
id_pettis(df)
```, caption: [Estimate of intrinsic dimensionality]) <4a_code2>

=== b)
#figure(```python
iterations = 100
n_generated = 100

noise_list = [[0, id_pettis(df)]]
df_noised = df.copy()
np.random.seed(1)
for i in range(1, iterations):
  x = np.random.uniform(min_max['x']['min'], min_max['x']['max'], n_generated//iterations)
  y = np.random.uniform(min_max['y']['min'], min_max['y']['max'], n_generated//iterations)
  z = np.random.uniform(min_max['z']['min'], min_max['z']['max'], n_generated//iterations)

  df_noised = pd.concat([df_noised, pd.DataFrame({'x': x, 'y': y, 'z': z})], ignore_index=True)
  noise_list.append([(n_generated//iterations)*i, id_pettis(df_noised)])
noise_df = pd.DataFrame(noise_list, columns=['n', 'eid'])

fig = px.scatter_3d(df_noised, x='x', y='y', z='z', color='z')
fig.show()

from plotnine import *
(
  ggplot(noise_df, aes(x='n', y='eid')) 
  + geom_line()
  + geom_hline(yintercept=eid, linetype='dashed', color='red')
  + geom_hline(yintercept=min(noise_df['eid']), linetype='dashed', color='blue')
  + labs(x='Number of Noise Data Points added', y='Estimate Intrinsic Dimensionality')
)
```, caption: [Noise generated and plotted])<4b_code>

== 5)
=== a)

#figure(```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

data = loadmat('../data/leukemia.mat')
X = data["leukemia"]
print("Dataset shape:", X.shape)

X_scaled = StandardScaler().fit_transform(X)

U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)

explained_variance = S**2 / np.sum(S**2)  # S contains singular values

plt.plot(range(1, len(S) + 1), explained_variance, marker='o', linestyle='-')
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.title("Scree Plot (Elbow Method)")

# find k
knee_locator = KneeLocator(range(1, len(S) + 1), explained_variance, curve="convex", direction="decreasing")
elbow_point = knee_locator.knee

# highlight elbow point
plt.axvline(x=elbow_point, color='r', linestyle='--', label=f'Elbow at {elbow_point}')
plt.scatter(elbow_point, explained_variance[elbow_point - 1], color='red', s=100, label='Elbow Point')
plt.legend()
plt.show()

k = 6 # reduce data to k=6
X_reduced_svd = U[:, :k] @ np.diag(S[:k])

# 2D plot
plt.figure(figsize=(7, 5))
plt.scatter(X_reduced_svd[:, 0], X_reduced_svd[:, 1], c='b', alpha=0.7)
plt.xlabel('First Component')
plt.ylabel('Second Component')
plt.title('2D Projection of Leukemia Data (SVD)')
plt.show()

# 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_reduced_svd[:, 0], X_reduced_svd[:, 1], X_reduced_svd[:, 2], c='r', alpha=0.7)
ax.set_xlabel('First Component')
ax.set_ylabel('Second Component')
ax.set_zlabel('Third Component')
ax.set_title('3D Projection of Leukemia Data (SVD)')
plt.show()
```, caption: [Singular value decomposition on leukemia dataset.])<5a_code>

=== b)

#figure(```python
import pyEDAkit as eda_lin

X_reduced_pca_cov = eda_lin.PCA(X_scaled, d=6, covariance=True, plot=False)

# 2D plot
plt.figure(figsize=(7, 5))
plt.scatter(X_reduced_pca_cov[:, 0], X_reduced_pca_cov[:, 1], c='g', alpha=0.7)
plt.xlabel('First Component')
plt.ylabel('Second Component')
plt.title('2D Projection of Leukemia Data (PCA - Covariance)')
plt.show()
```, caption: [Singular value decomposition on leukemia dataset.])<5b_code>

== 6)

#figure(```python
import numpy as np
import pyEDAkit as kit

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
A = genLDdata()
print(kit.IntrinsicDimensionality.MLE(A))
```, caption: [`genLDdata` function in python]) <genLDdata>

#figure(```python
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist

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
```, caption: [Local intrinsic calculation function (generated from _Example 2.10_)])

#figure(```python
import plotly.graph_objects as go
# Scatter plot with color map
colors = {1: 'red', 2: 'green', 3: 'blue', 4: 'black'}
fig = go.Figure()
labels = [1, 2, 3, 4]
for label in labels:
    indices = np.where(Ldim == label)[0]
    if len(indices) > 0:
        fig.add_trace(go.Scatter3d(
            x=A[indices, 0], y=A[indices, 1], z=A[indices, 2],
            mode='markers',
            marker=dict(color=colors[label], size=2),
            name=f"Dim {label}"
        ))

fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title="Intrinsic Dimensionality Scatterplot with [x]")
fig.show()
```, caption: [Local Intrinsic Dimensionality scatter plot])

#figure(```python
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
```, caption: [`genLDdata` modified]) <genLDdata_mod>

#figure(```python
A = genLDdata()

print("global intrinsic dimensionality (MLE):", kit.IntrinsicDimensionality.MLE(A))
Ldim, tabulation_df = compute_local_intrinsic_dimensionality(A, kit.IntrinsicDimensionality.MLE)
print(tabulation_df) # Ldim used for the plotting

print("global intrinsic dimensionality (CorrDim):", kit.IntrinsicDimensionality.corr_dim(A))
Ldim, tabulation_df = compute_local_intrinsic_dimensionality(A, kit.IntrinsicDimensionality.corr_dim)
print(tabulation_df)

A = genLDdata_mod()
print("global intrinsic dimensionality (PackingNumbers):", kit.IntrinsicDimensionality.packing_numbers(A))
Ldim, tabulation_df = compute_local_intrinsic_dimensionality(A, kit.IntrinsicDimensionality.packing_numbers)

```, caption: [Function calls for 6)])