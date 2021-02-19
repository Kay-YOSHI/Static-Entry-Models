function f = DoNPLLoop(theta, p_old_W, p_old_K)

% Variable declaration
global Wxmat Kxmat

% Monkey with err.tol this to see effects
err =10; 
err_tol = 1e-8;
NPL_reps = 0;
tot_reps = 2000;

while (err > err_tol)&&(NPL_reps < tot_reps)
    
    NPL_reps = NPL_reps + 1;
    old_theta = theta;
    
    % Minimize Pseudo Negative Log-likelihood
    options = optimoptions(@fminunc, 'display', 'iter', 'Algorithm', 'quasi-newton', 'MaxFunEvals', 10000);
    [kth_est, kth_fval, kth_exitflag, kth_output, kth_grad, kth_hess] = fminunc(@NPLnll, theta, options, p_old_W, p_old_K);
    
    % Update parameters
    theta = kth_est;
    NPL_thetaW = [theta(1) theta(3) theta(4) theta(5) theta(6) theta(7)]';
    NPL_thetaK = [theta(2) theta(3) theta(4) theta(5) theta(8)]';
    NPL_delta = exp(theta(9));
    
    % Compute Deterministic component of profits
    NPL_piW = Wxmat * NPL_thetaW;
    NPL_piK = Kxmat * NPL_thetaK;
    
    % Reconstruct probabilities
    p_new_W = normcdf(NPL_piW - NPL_delta .* p_old_K);
    p_new_K = normcdf(NPL_piK - NPL_delta .* p_old_W);
    
    % Acceleration Trick of Kasahara and Shimotsu
    p_new_W = normcdf(NPL_piW - NPL_delta .* p_new_K);
    p_new_K = normcdf(NPL_piK - NPL_delta .* p_new_W);
    
    % Compute error ||Pnew - Pold||
    err = transpose(p_new_K - p_old_K) * (p_new_K - p_old_K) / 2 + transpose(p_new_W - p_old_W) * (p_new_W - p_old_W) / 2;
    
    % Update probabilities
    p_old_W = p_new_W;
    p_old_K = p_new_K;
    
    %disp(NPL_reps);
    
    % Spit out INFO
    if err < err_tol
        NPL_est = kth_est;
        disp('Algorithm Converged.');
    end
end

f = NPL_est;
