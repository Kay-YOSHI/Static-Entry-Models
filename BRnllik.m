function f = BRnllik(zeta)

% Variable declaration
global xmat nf

% Setup parameters
k = size(xmat, 2);
theta = zeta(1:k, 1);
delta = exp(zeta(k+1, 1));

% Define Profits
Wprofit = xmat * theta;
Kprofit = Wprofit;

% Prob that duopoly profits are > 0
probWduo = normcdf(Wprofit - delta);
probKduo = probWduo;

% Prob of seeing duopoly
probduo = probWduo .* probKduo;

% Prob that monopoly profits are < 0
probWzero = normcdf(-Wprofit);
probKzero = probWzero;

% Prob of seeing no entrants
probzero = probWzero .* probKzero;

% Prob of seeing 1 store
probmon = 1 - probzero - probduo;

% Check for numerical issues
probzero(probzero <= 0) = 1e-10;
probmon(probmon <= 0) = 1e-10;
probduo(probduo <= 0) = 1e-10;

% Construct and return negative log likelihood
llik = ((nf==0) .* log(probzero)) + ((nf==1) .* log(probmon)) + ((nf==2) .* log(probduo));
f = -sum(llik);
