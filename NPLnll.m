function f = NPLnll(theta, p_old_W, p_old_K)

% Variable declaration
global Wxmat Kxmat Kmart WalMart

% Number of player specific parameters
kW = size(Wxmat, 2);
kK = size(Kxmat, 2);

% Setup parameters
thetaW = [theta(1) theta(3) theta(4) theta(5) theta(6) theta(7)]';
thetaK = [theta(2) theta(3) theta(4) theta(5) theta(8)]';
delta = exp(theta(9));

% Deterministic component of profits
piW = Wxmat * thetaW;
piK = Kxmat * thetaK;

%=========================================================================%
% Compute probabilities
%=========================================================================%
probW = normcdf(piW - delta .* p_old_K);
probK = normcdf(piK - delta .* p_old_W);

%=========================================================================%
% Construct and return Negative Log-Likelihood
%=========================================================================%
llik = WalMart .* log(probW) + (1-WalMart) .* log(1-probW) + Kmart .* log(probK) + (1-Kmart) .* log(1-probK);
f = -sum(llik);
