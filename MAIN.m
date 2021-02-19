%=========================================================================%
% Ellickson and Misra(2011) Matlab Code
%=========================================================================%
% Dataset : jiadata2R.mat
% The dataset has 2065 observations and 19 variables.
%
% Variables
%    county      : county identifier
%    population  : log of county population
%    SPC         : log of county retail sales per capita
%    urban       : percentage of urban population
%    MidWest     : dummy for MidWest region
%    dBenton     : log of distance to Benton county
%    southern    : dummy for the southern region
%    Kmart       : KMart stores (=1 if there is a Kmart store in the county)
%    WalMart     : Wal-Mart stores (=1 if there is a Kmart store in the county)
%    smallstores : number of small stores
%    dKmart      : distance weighted number of Kmart stores in markets that are not part of the sample
%    dWMart      : distance weighted number of Wal-Mart stores in markets that are not part of the sample
%    X1          : Optional
%    X2          : Optional
%    X3          : Optional
%    count0      : = 1 if nfirms=0, = 0 otherwise 
%    count1      : = 1 if nfirms=1, = 0 otherwise 
%    count2      : = 1 if nfirms=2, = 0 otherwise 
%    nfirms      : number of firms that actually entered the market
%=========================================================================%

% Variable declaration
global xmat nf Wxmat Kxmat nofirm Wm Km duo moverule WalMart Kmart predW predK

% Load "jiadata2R.mat"
load('jiadata2R.mat');

% Number of Observations
nmkts = 2065;

% Constant term
ints = ones(nmkts, 1);

% Starting values for estimation
%start = [-4.8960054 -15.3832747 1.6653983 0.9289151 2.4720639 -1.4612322 2.0311288 2.1255131 0.6889679]';
start = [0 0 0 0 0 0 0 0 0];

% Optiions for fminunc
options = optimoptions(@fminunc,'Display', 'Iter', 'Algorithm','quasi-newton', 'MaxFunEvals', 2000);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete Information - Model 1
%  - Bresnahan and Reiss(1991)
%    - Simple Number of Firms Estimator
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Observed Number of Firms
nf = nfirms;

% Relevant X matrix ( = market-level factors)
xmat = [ints, population, SPC, urban];

%=========================================================================%
% Minimize Negative-Log-Likelihood
%=========================================================================%

% fminunc ver.
[BR_est, BR_fval, BR_exitflag, BR_output, BR_grad, BR_hess] = fminunc(@BRnllik, zeros(5,1), options);

% fminsearch ver.
%options = optimset('Display','Iter', 'MaxFunEvals', 2000);
%[BR_est, BR_fval] = fminsearch(@BRnllik, zeros(5,1), options);

% Standard error of BR_est
BR_se = sqrt(diag(inv(BR_hess)));

% delta needs to be exponentiated
BR_delta_est = exp(BR_est(5));

% Standard error of delta
BR_delta_se = sqrt(exp(2 * BR_est(5))) * BR_se(5);

% RESULT
BR_est_result = [BR_est(2) BR_est(3) BR_est(4) BR_delta_est BR_est(1) 0 0 BR_est(1) 0]';
BR_se_result = [BR_se(2) BR_se(3) BR_se(4) BR_delta_se BR_se(1) 0 0 BR_se(1) 0]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete Information - Model 2, 3, 4
%  - Berry(1992) Estimators
%    - We implement three estimators
%      (i)   Based on assumption that most profitable firm moves first
%      (ii)  WalMart moves first 
%      (iii) Kmart moves first
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Covariates for Walmart and Kmart
Wxmat = [ints, population, SPC, urban, dBenton, southern];
Kxmat = [ints, population, SPC, urban, MidWest];

% Number of player specific parameters
kW = size(Wxmat, 2);
kK = size(Kxmat, 2);

nreps = 1000;
uW = randn(nmkts, nreps);
uK = randn(nmkts, nreps);

% Create dummy variables
duo = WalMart .* Kmart;       % =1 if duopoly
Wm = WalMart .* (1 - Kmart);  % =1 if WalMart monopoly
Km = (1 - WalMart) .* Kmart;  % =1 if Kmart monopoly
nofirm = 1 - duo - Wm - Km;   % =1 if no firm enter

%=========================================================================%
% Estimation for each equilibrium selection approach                      %
%=========================================================================%

% (i) Most profitable firm moves first
moverule = 'profit';
[Berry_P_est, Berry_P_fval, Berry_P_exitflag, Berry_P_output, Berry_P_grad, Berry_P_hess] = fminunc(@BerryObj, start, options);

% (ii) WalMart moves first
moverule = 'WalMart';
[Berry_W_est, Berry_W_fval, Berry_W_exitflag, Berry_W_output, Berry_W_grad, Berry_W_hess] = fminunc(@BerryObj, start, options);

% (iii) Kmart moves first
moverule = 'Kmart';
[Berry_K_est, Berry_K_fval, Berry_K_exitflag, Berry_K_output, Berry_K_grad, Berry_K_hess] = fminunc(@BerryObj, start, options);

% Standard Errors
Berry_P_se = sqrt(diag(inv(Berry_P_hess)));
Berry_W_se = sqrt(diag(inv(Berry_W_hess)));
Berry_K_se = sqrt(diag(inv(Berry_K_hess)));

% delta needs to be exponentiated
Berry_P_delta_est = exp(Berry_P_est(9));
Berry_W_delta_est = exp(Berry_W_est(9));
Berry_K_delta_est = exp(Berry_K_est(9));

% Standard error of delta
Berry_P_delta_se = sqrt(exp(2 * Berry_P_est(9))) * Berry_P_se(9);
Berry_W_delta_se = sqrt(exp(2 * Berry_W_est(9))) * Berry_W_se(9);
Berry_K_delta_se = sqrt(exp(2 * Berry_K_est(9))) * Berry_K_se(9);

% Parameter Estimates
Berry_P_est_result = [Berry_P_est(3) Berry_P_est(4) Berry_P_est(5) Berry_P_delta_est, ...
                      Berry_P_est(1) Berry_P_est(6) Berry_P_est(7) Berry_P_est(2) Berry_P_est(8)]';
Berry_W_est_result = [Berry_W_est(3) Berry_W_est(4) Berry_W_est(5) Berry_W_delta_est, ...
                      Berry_W_est(1) Berry_W_est(6) Berry_W_est(7) Berry_W_est(2) Berry_W_est(8)]';
Berry_K_est_result = [Berry_K_est(3) Berry_K_est(4) Berry_K_est(5) Berry_K_delta_est, ...
                      Berry_K_est(1) Berry_K_est(6) Berry_K_est(7) Berry_K_est(2) Berry_K_est(8)]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Incomplete Information - Model 1
%  - Nested Fixed Point Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Minimize Negative Log-likelihood
[NFXP_est, NFXP_fval, NFXP_exitflag, NFXP_output, NFXP_grad, NFXP_hess] = fminunc(@NFXPnll, start, options);

% delta needs to be exponentiated
NFXP_delta_est = exp(NFXP_est(9));

% Standard Error of NFXP_est
NFXP_se = sqrt(diag(inv(NFXP_hess)));

% Standard error of delta
NFXP_delta_se = sqrt(exp(2 * NFXP_est(9))) * NFXP_se(9); 

% RESULT
NFXP_est_result = [NFXP_est(3) NFXP_est(4) NFXP_est(5) NFXP_delta_est, ...
                      NFXP_est(1) NFXP_est(6) NFXP_est(7) NFXP_est(2) NFXP_est(8)]';
NFXP_se_result = [NFXP_se(3) NFXP_se(4) NFXP_se(5) NFXP_delta_se, ...
                      NFXP_se(1) NFXP_se(6) NFXP_se(7) NFXP_se(2) NFXP_se(8)]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Incomplete Information - Model 2
%  - Two Step approach
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%=========================================================================%
% First Step : Estimate CCPs by Logistic Regression
%=========================================================================%

% Construct CCPs : Logistic or Probit Regression
WxLogit = [population, SPC, urban, dBenton, southern];
KxLogit = [population, SPC, urban, MidWest];

% Logistic Regression ver.
LogitW = glmfit(WxLogit, WalMart, 'binomial', 'logit');
LogitK = glmfit(KxLogit, Kmart, 'binomial', 'logit');
predW = glmval(LogitW, WxLogit, 'logit'); 
predK = glmval(LogitK, KxLogit, 'logit');

% Probit Regression ver.
%ProbitW = glmfit(WxLogit, WalMart, 'binomial', 'probit');
%ProbitK = glmfit(KxLogit, Kmart, 'binomial', 'probit');
%predW = glmval(LogitW, WxLogit, 'probit'); 
%predK = glmval(LogitK, KxLogit, 'probit');

%=========================================================================%
% Estimation                                                              %
%=========================================================================%

% Minimize Negative Log-likelihood
[TS_est, TS_fval, TS_exitflag, TS_output, TS_grad, TS_hess] = fminunc(@TwoStepnll, start, options);

% delta needs to be exponentiated
TS_delta_est = exp(TS_est(9));

%=========================================================================%
% Two Step Approach_Standard Errors : Bootstrap                           %
%=========================================================================%
%{
% Setup for Bootstrap
bootN = 100;

theta = TS_est;
TSBoot_est = zeros(bootN, size(theta, 2));
datamat = [WalMart, Kmart, population, SPC, urban, dBenton, southern, MidWest];

% Bootstrap
for b = 1:bootN
    
    % Array for new data
    TSDataBoot = [];
    
    % Generate data for bootstrap
    ID = randsample(nmkts, nmkts, true);
    TSDataBoot = datamat(ID, :);
    
    % Generate variables
    nmktsBoot = size(TSDataBoot, 1);
    intsBoot = ones(nmktsBoot, 1);
    
    % Independent Variable
    Wxmat = [intsBoot, TSDataBoot(:, 3), TSDataBoot(:, 4), TSDataBoot(:, 5), TSDataBoot(:, 6), TSDataBoot(:, 7)];
    Kxmat = [intsBoot, TSDataBoot(:, 3), TSDataBoot(:, 4), TSDataBoot(:, 5), TSDataBoot(:, 8)];
    
    % Dependent Variable
    WalMart = TSDataBoot(:, 1);
    Kmart = TSDataBoot(:, 2);
    
    % Execute TwoStepnll.m using new data
    [TS_est, TS_fval, TS_exitflag, TS_output, TS_grad, TS_hess] = fminunc(@TwoStepnll, theta, options);
    
    % Store estimated parameters
    TSBoot_est(b, :) = TS_est;
    
end

% Standard Error
TS_se = std(TSBoot_est);

% Standard error of delta
TS_delta_se = std(exp(TSBoot_est(:, 9))); 
%}

% RESULT
TS_est_result = [TS_est(3) TS_est(4) TS_est(5) TS_delta_est, ...
                      TS_est(1) TS_est(6) TS_est(7) TS_est(2) TS_est(8)]';
%TS_se_result = [TS_se(3) TS_se(4) TS_se(5) TS_delta_se, ...
%                      TS_se(1) TS_se(6) TS_se(7) TS_se(2) TS_se(8)]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Incomplete Information - Model 3
%  - Nested Pseudo Likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Starting CCPs
p_old_W = WalMart;
p_old_K = Kmart;

% NPL Loop
NPL_est = DoNPLLoop(start, p_old_W, p_old_K);
NPL_delta_est = exp(NPL_est(9));

%=========================================================================%
% NPL_Standard Errors : Bootstrap                                         %
%=========================================================================%
%{
% Setup for Bootstrap
theta = NPL_est;
NPLBoot_est = zeros(bootN, size(theta, 2));
datamat = [WalMart, Kmart, population, SPC, urban, dBenton, southern, MidWest];

% Bootstrap
for b = 1:bootN
    
    % Array for new data
    NPLDataBoot = [];
    
    % Generate data for bootstrap
    ID = randsample(nmkts, nmkts, true);
    NPLDataBoot = datamat(ID, :);
    
    % Generate variables
    nmktsBoot = size(NPLDataBoot, 1);
    intsBoot = ones(nmktsBoot, 1);
    
    % Independent Variable
    Wxmat = [intsBoot, NPLDataBoot(:, 3), NPLDataBoot(:, 4), NPLDataBoot(:, 5), NPLDataBoot(:, 6), NPLDataBoot(:, 7)];
    Kxmat = [intsBoot, NPLDataBoot(:, 3), NPLDataBoot(:, 4), NPLDataBoot(:, 5), NPLDataBoot(:, 8)];
    
    % Dependent Variable
    WalMart = NPLDataBoot(:, 1);
    Kmart = NPLDataBoot(:, 2);
    
    % Starting CCPs
    p_old_W = WalMart;
    p_old_K = Kmart;
    
    NPL_est = DoNPLLoop(theta, p_old_W, p_old_K);
    
    % Store estimated parameters
    NPLBoot_est(b, :) = NPL_est;
    
end

% Standard Error
NPL_theta_se = std(NPLBoot_est);

% Standard error of delta
NPL_delta_se = std(exp(NPLBoot_est(:, 9))); 
%}

% RESULT
NPL_est_result = [NPL_est(3) NPL_est(4) NPL_est(5) NPL_delta_est, ...
                      NPL_est(1) NPL_est(6) NPL_est(7) NPL_est(2) NPL_est(8)]';
%NPL_se_result = [NPL_theta_se(3) NPL_theta_se(4) NPL_theta_se(5) NPL_delta_se, ...
%                      NPL_theta_se(1) NPL_theta_se(6) NPL_theta_se(7) NPL_theta_se(2) NPL_theta_se(8)]';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                Result                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
horz0 = ['   BR(1991)  Berry(i)  Berry(ii) Berry(iii)  NFXP    TwoStep     NPL'];
vert=['Population               ';
      'Retail Sales per Capita  ';
      'Urban                    ';
      'Delta                    ';
      'Intercept(WalMart)       ';
      'Dist to Bentonville, AK  ';
      'South                    ';
      'Intercept(Kmart)         ';
      'Midwest                  '];

disp('  ')
disp('Parameter Estimate')
disp('================================================================')
disp(horz0)
disp('================================================================')
disp('  ')

for i=1:size(vert,1)
     disp(vert(i,:))
     disp([BR_est_result(i), Berry_P_est_result(i), Berry_W_est_result(i), Berry_K_est_result(i), NFXP_est_result(i), TS_est_result(i), NPL_est_result(i)])
end
disp('================================================================')
%=========================================================================%