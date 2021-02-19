function f = NFXPnll(theta)

% Variable declaration
global Wxmat Kxmat Kmart WalMart

% Setup parameters
thetaW = [theta(1) theta(3) theta(4) theta(5) theta(6) theta(7)]';
thetaK = [theta(2) theta(3) theta(4) theta(5) theta(8)]';
delta = exp(theta(9));

% Deterministic component of profits
piW = Wxmat * thetaW;
piK = Kxmat * thetaK;

%=========================================================================%
% Nested Fixed Point Computation
%=========================================================================%

nfxpreps = 0;
err = 10;

p_old_K = Kmart;
p_old_W = WalMart;

while (err > 1e-12)&&(nfxpreps < 1000)
    nfxpreps = nfxpreps + 1;
    p_new_W = normcdf(piW - delta .* p_old_K);
    p_new_K = normcdf(piK - delta .* p_new_W);
    err = max( abs(p_new_K - p_old_K) + abs(p_new_W - p_old_W) );
    p_old_W = p_new_W;
    p_old_K = p_new_K;
end

%=========================================================================%
% Construct and return Negative Log-Likelihood
%=========================================================================%
llik = WalMart .* log(p_old_W) + (1-WalMart) .* log(1-p_old_W) + Kmart .* log(p_old_K) + (1-Kmart) .* log(1-p_old_K);
f = -sum(llik);
