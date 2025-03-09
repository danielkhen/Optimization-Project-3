function [lambda, w, w0] = AugmentedLagrangianSVM(X, y, C, p0, p_max, mu0, p_mult, tol, max_iter, Flag, sigma, beta, alpha0)
    % Augmented Lagrangian method for solving the soft margin SVM problem
    %
    % Inputs:
    % X - Training data (N x d matrix)
    % y - Labels (N x 1 vector)
    % C - Regularization parameter
    % p0 - Initial penalty parameter
    % p_max - Maximum penalty
    % mu0 - Initial Lagrange multiplier estimate
    % p_mult - Penalty increase multiplier
    % tol - Convergence tolerance for both inner and outer algorithms.
    % max_iter - Maximum iterations for both inner and outer loops
    % Flag - 'projGrad' for projected gradient, 'projNewton' for projected Newton
    % sigma   - Parameter for Armijo condition (0 < sigma < 1)
    % beta    - Backtracking parameter (0 < beta < 1)
    % alpha0  - Initial step size
    %
    % Output:
    % lambda_opt - Optimized dual variables
    [N, d] = size(X);
    lambda = zeros(N, 1);  % Initialize lambda
    p = p0;
    mu = mu0;
    Q = (diag(y) * X) * (X' * diag(y));  % Compute the Q matrix (hessian of soft margin SVM)
    
    for k = 1:max_iter
        % Solve inner problem using Projected Newton
        lambda_prev = lambda;
        lambda = ProjectedNewton(lambda, Q, y, C, p, mu, tol, Flag, sigma, beta, alpha0);
        
        % Update Lagrange multiplier
        g_lambda = sum(lambda .* y);
        mu = 2 * p * g_lambda + mu;
        
        % Update penalty parameter
        p = min(p_mult * p, p_max);
        
        % Convergence
        if norm(lambda - lambda_prev) < tol
            fprintf("outer convergence after %i iterations\n", k);
            break;
        end
    end

    % Compute optimal w
    w = sum((lambda .* y) .* X, 1)';
    
    % Identify support vectors (0 < lambda < C)
    support_indices = find(lambda > 0 & lambda < C);
    
    % Compute estimated bias w0
    if ~isempty(support_indices)
        w0 = mean(y(support_indices) - X(support_indices, :) * w);
    else
        w0 = 0;
    end
end

function lambda = ProjectedNewton(lambda, Q, y, C, p, mu, tol, Flag, sigma, beta, alpha0)
    % Projected Newton method to solve the constrained optimization
    %
    % Inputs:
    % lambda - Initial guess
    % Q - Hessian matrix for dual SVM
    % y - Label vector
    % C - Regularization parameter
    % p - Current penalty parameter
    % mu - Current Lagrange multiplier
    % tol - Convergence tolerance
    % Flag - 'projGrad' for projected gradient, 'projNewton' for projected Newton
    % sigma   - Parameter for Armijo condition (0 < sigma < 1)
    % beta    - Backtracking parameter (0 < beta < 1)
    % alpha0  - Initial step size
    %
    % Output:
    % lambda_opt - Optimized lambda

    epsilon = 1e-6;  % Small regularization term
    H = Q + 2 * p * (y * y') + epsilon * eye(length(lambda));
    grad_F = @(lambda) Q * lambda - ones(size(lambda)) + (2 * p * sum(lambda .* y) + mu) * y;
    
    max_inner_iter = 100;
    for k = 1:max_inner_iter
        grad = grad_F(lambda);
        d = -grad;
        
        if strcmp(Flag, 'projNewton')
            % Compute active set: Indices where lambda is at 0 or C
            active_set = (lambda <= 0 & grad_F(lambda) > 0) | (lambda >= C & grad_F(lambda) < 0);
            free_set = ~active_set;
            
            % Compute reduced Hessian
            H_R = H(free_set, free_set);
            grad_R = grad(free_set);
            
            % Compute Newton step for free variables
            d_R = -(H_R \ grad_R);  % Solve linear system
            d(free_set) = d_R;
        end

        % Perform Armijo line search
        alpha = ArmijoRule(@(l) F_p_mu(l, Q, y, p, mu), lambda, F_p_mu(lambda, Q, y, p, mu), grad_F(lambda), d, sigma, beta, alpha0, "boxConstraints", C);
        
        % Update and project onto feasible region
        lambda_prev = lambda;
        lambda = max(0, min(C, lambda + alpha * d));
        
        % Convergence
        if norm(lambda - lambda_prev) < tol
            fprintf("inner convergence after %i iterations\n", k);
            break;
        end
    end
end

function F = F_p_mu(lambda, Q, y, p, mu)
    % Compute the augmented Lagrangian objective function
    g_lambda = sum(lambda .* y);
    F = 0.5 * lambda' * Q * lambda - sum(lambda) + p * g_lambda^2 + mu * g_lambda;
end