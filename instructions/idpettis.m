function idhat = idpettis(ydists,n)
%IDPETTIS    ESTIMATE OF INTRINSIC DIMENSIONALITY
%
%   IDHAT = IDPETTIS(YDIST,N)
%   Returns the estimate of intrinsic dimensionality using the algorithm of
%   Pettis, Bailey, Jain, and Dubes, 1979. It is based on the
%   nearest-neighbor distances of a data set.
%
%   The input argument YDIST contains the nearest neighbor distances as
%   obtained by the Statistics Toolbox function PDIST. It is a vector
%   containing the pair-wise distances. N is the sample size.

% Written July 2003.
K = 5;
% Do the first one.
tmp = sort(ydists(1:(n-1)));
kdist = zeros(n,K);
kdist(1,:) = tmp(1:K);
for i = 2:n
    tmp = zeros(1,n);
    J = (i+1):n;
    tmp(J) = ydists((i-1).*(n-i/2)+J-i);
    I = 1:(i-1);
    tmp(I) = ydists((I-1).*(n-I/2)+i-I);
    tmp = sort(tmp(:));
    kdist(i,:) = tmp(2:(K+1))';
end

% kmax corresponds to the last column
mmax = mean(kdist(:,K));
smax = sqrt(var(kdist(:,K)));
k = 1:K;

% Get the averages for the estimate
% but remove the outliers - put into a cell array
for i=1:K
    ind = find(kdist(:,i) <= (mmax + smax));
    kcell{i} = kdist(ind,i);
    logrk(i) = log(mean(kcell{i}));
end

% get initial value for d
logk = log(k);
[p,s] = polyfit(logk,logrk,1);
dhat = 1/p(1);
dhatold = realmax;
maxiter = 100;
epstol = 0.01;
i = 0;
while abs(dhatold - dhat) >= epstol & i < maxiter
    % Adjust the y values by adding logGRk
    logGRk = (1/dhat)*log(k)+gammaln(k)-gammaln(k+1/dhat);
    [p,s] = polyfit(logk,logrk + logGRk,1);
    dhatold = dhat;
    dhat = 1/p(1);
    i = i+1;
end
idhat = dhat;
