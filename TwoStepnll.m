function f = TwoStepnll(theta)

% Global Variables
global Wxmat Kxmat Kmart WalMart predW predK

% Setup parameters
thetaW = [theta(1) theta(3) theta(4) theta(5) theta(6) theta(7)]';
thetaK = [theta(2) theta(3) theta(4) theta(5) theta(8)]';
delta = exp(theta(9));

% Deterministic component of profits
piW = Wxmat * thetaW;
piK = Kxmat * thetaK;

%=========================================================================%
% Compute probabilities : Second Step
%=========================================================================%

probW = normcdf(piW - delta .* predK);
probK = normcdf(piK - delta .* predW);

%=========================================================================%
% Construct and return Negative Log-Likelihood
%=========================================================================%
llik = WalMart .* log(probW) + (1-WalMart) .* log(1-probW) + Kmart .* log(probK) + (1-Kmart) .* log(1-probK);
f = -sum(llik);
