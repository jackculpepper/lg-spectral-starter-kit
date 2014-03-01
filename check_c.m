
%% set some params
M = 15;        % number of transformation basis functions
L = 81;        % total pixels being transformed 9x9 patch -> 81 pixels
T = 4;         % number of time points to fit (in the paper, T=2)
lambda = 0;    % strength of sparseness penalty
lambda = 1;    % strength of sparseness penalty

%% initialize to positive numbers
%% avoids discontinuity at zero while testing the gradient
c = 0.1*rand(M, 1);
psi = rand(L, L, M);

%% normalize each transformation basis function
%% reduces the possibility of numeric overflow while testing the gradient
for i = 1:M
    psi(:,:,i) = psi(:,:,i)*diag(1./sqrt(sum(sum(psi(:,:,i).^2))));
end

%% generate a random image sequence
I = randn(L, T);

tic
checkgrad('objfun_c', c(:), 1e-4, psi, I, lambda)
toc

