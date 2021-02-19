function f = BerryObj(theta)

% Variable declaration
global Wxmat Kxmat nofirm Wm Km duo moverule

% Setup parameters
thetaW = [theta(1) theta(3) theta(4) theta(5) theta(6) theta(7)]';
thetaK = [theta(2) theta(3) theta(4) theta(5) theta(8)]';
delta = exp(theta(9));

% Deterministic component of profits
piW = Wxmat * thetaW;
piK = Kxmat * thetaK;

%=========================================================================%
% Analytical probabilities
%=========================================================================%

% Entry under assumption that WalMart moves first
if strcmp(moverule, 'WalMart') == 1
    p0an = normcdf(-piW) .* normcdf(-piK);
    p2an = normcdf(piW - delta) .* normcdf(piK - delta);
    pWan = normcdf(piW) .* normcdf(-piK + delta);
    pKan = 1 - p0an - p2an - pWan;
end

% Entry under assumption that Kmart moves first
if strcmp(moverule, 'Kmart') == 1
    p0an = normcdf(-piW) .* normcdf(-piK);
    p2an = normcdf(piW - delta) .* normcdf(piK - delta);
    pKan = normcdf(piK) .* normcdf(-piW + delta);
    pWan = 1 - p0an - p2an - pKan;
end

% Entry under assumption that the more profitable firm moves first
if strcmp(moverule, 'profit') == 1
    p0an = normcdf(-piW) .* normcdf(-piK);
    p2an = normcdf(piW - delta) .* normcdf(piK - delta);
    %pWan = normcdf(piW) .* normcdf(-piK + delta) - ( normcdf(-piW + delta) - normcdf(-piW) ) .* ( normcdf(-piK + delta) - normcdf(-piK) ) .* ( 1 - normcdf( (piK - piW)/2 ) );
    pWan = normcdf(piW) .* normcdf(-piK + delta) - ( normcdf(-piW + delta) - normcdf(-piW) ) .* ( normcdf(-piK + delta) - normcdf(-piK) ) .* ( 1 - normcdf( (piW - piK)/2 ) );
    pKan = 1 - p0an - p2an - pWan;
end

%=========================================================================%
% Construct and return Negative Log-Likelihood
%=========================================================================%

% Analytical probabilities
psimDuo = p2an;
psimW = pWan;
psimK = pKan;
psim0 = p0an;

llik0 = nofirm .* log(psim0) + duo .* log(psimDuo) + Wm .* log(psimW) + Km .* log(psimK);
llik1 = sum(llik0);
f = -sum(llik1);
