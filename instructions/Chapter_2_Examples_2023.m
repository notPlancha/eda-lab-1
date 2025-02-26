% Exploratory data analysis with matlab BOOK
% chapter Two: Dimension reduction: Linear
addpath( 'C:\Users\tsyuch\Desktop\Utbildningar\Masterutbildning\Data science\MA661E\Matlab_Code_Book\EDA Toolbox V3' );
%%  Example 2.2  Cell cycle dataset 384 genes and measured 17 times 
addpath( 'C:\Users\tsyuch\Desktop\Utbildningar\Masterutbildning\Data science\MA661E\Matlab_Code_Book\EDA Toolbox V3' );
load yeast
[n,p] = size(data); % should be 384 x 17 
datac = data - repmat(sum(data)/n,n,1);% Center the data.
covm = corr(datac);% Find the covariance matrix. 
[eigvec,eigval] = eig(covm);
eigval = diag(eigval); % Extract the diagonal elements
eigval = flipud(eigval);  % Order in descending order
eigvec = eigvec(:,p:-1:1);
figure, plot(1:length(eigval),eigval,'ko-') %  Scree plot. 
title('Scree Plot')
xlabel('Eigenvalue Index - k')
ylabel('Eigenvalue')
pervar = 100*cumsum(eigval)/sum(eigval);% Now for the percentage of variance explained.
% First get the expected sizes of the eigenvalues.
g = zeros(1,p);
for k = 1:p
for i = k:p
g(k) = g(k) + 1/i;
end
end
g = g/p;
propvar = eigval/sum(eigval);
g(1:4) 
propvar(1:4)
avgeig = mean(eigval);  % Now for the size of the variance.
ind = find(eigval > avgeig); % Find the length of ind:
length(ind)
P = eigvec(:,1:3); % Using d = 3, we will reduce the dimensionality.
Xp = datac*P;
figure,plot3(Xp(:,1),Xp(:,2),Xp(:,3),'k*')
xlabel('PC 1'),ylabel('PC 2'),zlabel('PC 3')
%% Example 2.4
load lsiex  %  Loads up variable: X, termdoc, docs, and words
[n,p] = size(termdoc); % 6 terms and 5 docs,  6 x 5 
% Normalize columns to be unit norm
 for i = 1:p
 termdoc(:,i) = X(:,i)/norm(X(:,i));
 end
[W,H] = nnmf(termdoc,3,'algorithm', 'mult') % 'als') %'mult')
 q1 = [1 0 1 0 0 0]';
 q2 = [1 0 0 0 0 0]';
 % Find the cosine of the angle between
% columns of termdoc and a query vector.
% Note that the magnitude of q1 is not 1.
m1 = norm(q1);% Note that the magnitude of q1 is not 1.
cosq1a = q1'*termdoc/m1% Find the cosine of the angle between columns of termdoc and a query vector.
% The magnitude of q2 happens to be 1.
cosq2a = q2'*termdoc
%% Eample 2.5 Stocks share  development in procent for 10 companies during 200 days
load stockreturns % Loads up a variable called stocks.
lab={'1', '2', '3', '4', '5','6', '7', '8', '9', '10'};
t=linspace(-1, 1, 20);
[LamVrot,PsiVrot] = factoran(stocks,3);% Perform factor analysis: 3 factors,default rotation.
plot(LamVrot(:, 1), LamVrot(:, 2), '.', t, 0*t, 'b',  0*t, t, 'b')
 text(LamVrot(:, 1)+0.02, LamVrot(:, 2), lab)
xlabel('Factor 1')
ylabel('Factor 2')
title('\bf Default')
 
[Lam,Psi] = factoran(stocks,3,'rotate','none');% without rotation
figure % Non rotation
plot(Lam(:, 1), Lam(:, 2), '.', t, 0*t, 'b',  0*t, t, 'b')
text(Lam(:, 1)+0.02, Lam(:, 2), lab)
xlabel('Factor 1')
ylabel('Factor 2')
title('\bf No rotation')
 
[LProt,PProt, T, Stat, F]=factoran(stocks,3,'rotate','promax');% With the promax rotation.
figure % Promax rotation
size(LProt)
plot(LProt(:, 1), LProt(:, 2), '.', t, 0*t, 'b',  0*t, t, 'b')
text(LProt(:, 1)+0.02, LProt(:, 2), lab)
xlabel('Factor 1')
ylabel('Factor 2')
title('\bf Promax rotation')

% Little extra
% reduced  in 3-dim
X3=F*LProt';
subplot(3, 1, 1)
plot(X3(:,1), X3(:,2), '*')
xlabel('Component 1')
ylabel('Component 2')
subplot(3, 1, 2)
plot(X3(:,2), X3(:,3), '*')
xlabel('Component 2')
ylabel('Component 3')
subplot(3, 1, 3)
plot(X3(:,3), X3(:,1), '*')
xlabel('Component 3')
ylabel('Component 1')
%plot3(X3(:,1), X3(:,2), X3(:,3), '*')


%% Example 2.6 (page 60) New in third edition
% Linear discrimanant analysis (LDA)
n1=100;
n2=100;
cov1=eye(2);
cov2=[1 0.7; 0.7, 1]; % Original value 0.9
dat1=mvnrnd([-2 2], cov1, n1);
dat2=mvnrnd([2 -2], cov2, n2);
plot(dat1(:,1),dat1(:,2), 'x',dat2(:,1), dat2(:,2), 'o') % figure 2.7 a
 figure
scat1=(n1-1)*cov(dat1);
scat2=(n2-1)*cov(dat2);
Sw= scat1 +scat2;
mu1=mean(dat1);
mu2=mean(dat2);
w= inv(Sw)*(mu1'-mu2');
w=w/norm(w);
pdat1=w'*dat1';
pdat2=w'*dat2';
[ksd1, x1]=ksdensity(pdat1);
[ksd2, x2]=ksdensity(pdat2);
plot(pdat1, zeros(1, 100), 'x', pdat2, zeros(1, 100), 'o');
hold on 
plot(x1, ksd1, x2, ksd2) % figure 2.7 b
hold off

%% Example 2.8 Intrinsic dimension
% Generate the random numbers
% unifrnd is from the Statistics Toolbox.
n = 500;
theta0 = linspace(0,4*pi,n);
% Use in the equations for a helix.
x0 = cos(theta0);
y0 = sin(theta0);
z0 = 0.1*(theta0);
plot3(x0, y0, z0) % orignal helix curve
figure 
n = 500;
theta = unifrnd(0,4*pi,1,n); % random order
% Use in the equations for a helix.
x = cos(theta);
y = sin(theta);
z = 0.1*(theta);
plot3(x, y, z, '*') % randomized helix curve
% Put into a data matrix.
X = [x(:),y(:),z(:)];
ydist = pdist(X);
d = idpettis(ydist,n) % 1.0684
  
%% Example 2.9  page 75
% Generate helix curve
[X]=generate_data('helix', 2000, 0.05);
plot3(X(:, 1), X(:, 2), X(:, 3), '.')
grid on 
d_corr=intrinsic_dim(X, 'CorrDim') % 1,6706
d_mle=intrinsic_dim(X, 'MLE')% 1.6951
d_pack=intrinsic_dim(X, 'PackingNumbers') % 1.2320

%% Example 2.10  page 77
A=genLDdata;
% estimate the global intrisic dimensionality, default is MLE
Dg = intrinsic_dim(A) % global dimensionality 1.5229
% get the pairwise interpoint distance
% use the default Euclidean distance
Ad =squareform(pdist(A));
% get the dimension of data
[nr, nc]=size(A);
Ldim=zeros(nr, 1);
Ldim2 = Ldim;
[Ads, J]= sort(Ad, 2);
% set the neighborhood size
k=100;% orignal 100
for m=1:nr;
    Ldim(m, 1)= ...
        intrinsic_dim(A(J(m,1:k), :));
end
% local dimension
Ldim(Ldim>3)=4;
Ldim=ceil(Ldim);
% Tabulate them
tabulate(Ldim) 
% Scatterplot with color map
ind1= find(Ldim == 1);
ind2= find(Ldim == 2);
ind3= find(Ldim == 3);
ind4= find(Ldim == 4);
scatter3(A(ind1, 1), A(ind1, 2),A(ind1, 3), 'r.')
hold on
scatter3(A(ind2, 1), A(ind2, 2),A(ind2, 3), 'g.')
scatter3(A(ind3, 1), A(ind3, 2),A(ind3, 3), 'b.')
scatter3(A(ind4, 1), A(ind4, 2),A(ind4, 3), 'k.')
hold off

%% SVD application to clown 
% page 154 in Modern multidimensional scaling
load clown
size(X)
[U, D, V]=svd(X);
image(X)
colormap(map)
title('Clown: Original')
% dim 10 approximation
figure
U10=U(:, 1:10);
D10=D(1:10, 1:10);
V10=V(:, 1:10);
size(U10)
X10= (U10)*(D10)*(V10)';
image(X10)
colormap(map)
title('Clown: 10-dim approximation')
% dim 20 approximation
figure 
U20=U(:, 1:20);
D20=D(1:20, 1:20);
V20=V(:, 1:20);
X20= U20*D20*V20';
image(X20)
colormap(map)
title('Clown: 20-dim approximation')
