function [f,g] = objfun_c(x0, psi, I, lambda);

[L T] = size(I);
Lsz = sqrt(L);

c = x0;
M = size(c, 1);

%% compose the matrix exponent
A = zeros(L);
for i = 1:M
    A = A + psi(:,:,i) * c(i);
end

%% compute the spectral decomposition of the exponent
[U D] = eig(A);
V = inv(U).';
D = diag(D);

%% use Matlab's built-in method to exponentiate A
ExpA = expm(A);


%% compute the residual's contribution to E
%% ..but only compute the portion that depends on c
f1 = 0; f2 = 0;

for t = 1:T-1
    f1 = f1 - I(:,t+1)'*ExpA*I(:,t);
    f2 = f2 + 0.5*I(:,t)'*ExpA'*ExpA*I(:,t);
end

%% compute the sparse penalty on c
f3 = lambda*sum(abs(c(:)));


%% sum the 1st and 2nd order residual terms with the sparse penalty
f = f1+f2+f3;



%% populate F
F = zeros(L,L);

ExpD = exp(D);
for i = 1:L
    for j = 1:L
        if D(i) == D(j)
            F(i,j) = ExpD(i);
        else
            F(i,j) = (ExpD(j) - ExpD(i)) / (D(j) - D(i));
        end
    end
end

%% reshape psi so we can compute the double sum in eqn 10 as a matrix mult
psi_2d = reshape(psi, L^2, M);

%% accumulate dE/dc across time
%% corresponds to eqns 7-10 of the paper
%% (there is a minus sign missing in the paper!)
dc = zeros(1,M);

for t = 1:T-1
    P = U.' * (-I(:,t+1)*I(:,t)' + (I(:,t)*I(:,t)'*ExpA')') * V;
    %%         ^
    %% the missing sign

    Q = V * (F .* P) * U.';

    dc = dc + Q(:)' * psi_2d;
end

%% add in the derivative of the sparse penalty
dc = dc + lambda*sign(c)';

%% make sure only the real part comes through
%% rounding errors can cause a small imaginary part to be present in dc
g = real(dc(:));

