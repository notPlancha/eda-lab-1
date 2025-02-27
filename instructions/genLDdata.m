function A = genLDdata
% This function will return 1,000 points
% in 3 dimensions.
%
%   X = genLDdata
%
%   It consists of data along two lines,
%   data on the surface of a sphere, and
%   data in a cube. This data set can be
%   used to explore the estimation of
%   local dimension.

 %sample from the surface of a sphere
 X1 = randn(1000,1);
 X2 = randn(1000,1);
 X3 = randn(1000,1);
 lambda = sqrt(X1.^2 + X2.^2 + X3.^2);
 X1 = X1./lambda;
 X2 = X2./lambda;
 X3 = X3./lambda;
 X = [X1,X2,X3];
 plot3(X(:,1),X(:,2),X(:,3),'r.')
 
%sample from a cube
 X1 = rand(1000,1) + 2;
 X2 = rand(1000,1) + 2;
 X3 = rand(1000,1) + 2;
 
 XX = [X1,X2,X3];
 hold on
 plot3(XX(:,1),XX(:,2),XX(:,3),'g.')
 
 %sample from lines attached to a sphere
 X1 = zeros(1000,1);
 X2 = zeros(1000,1);
 X3 = 2*rand(1000,1) + 1;
 L1 = [X1,X2,X3];
 
 X1 = zeros(1000,1);
 X2 = zeros(1000,1);
 X3 = -2*rand(1000,1) + -1;
 L2 = [X1,X2,X3];
 
 X1 = zeros(1000,1);
 X2 = 2*rand(1000,1)+1;
 X3 = zeros(1000,1);
 L3 = [X1,X2,X3];
 
 X1 = zeros(1000,1);
 X2 = -2*rand(1000,1)-1;
 X3 = zeros(1000,1);
 L4 = [X1,X2,X3];
 
 A = zeros(6000,3);
 A(1:1000,:) = X;
 A(1001:2000,:) = XX;
 A(2001:3000,:) = L1;
 A(3001:4000,:) = L2;
 A(4001:5000,:) = L3;
 A(5001:6000,:) = L4;
 
 plot3(A(:,1),A(:,2),A(:,3),'b.')
 grid on
 box on
 
 hold off
