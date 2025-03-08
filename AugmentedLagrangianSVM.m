function [w, w0] = AugmentedLagrangianSVM(X, y, C, p0, p_max, mu0, beta, tol, max_iter)
    % Augmented Lagrangian method for solving the dual SVM problem
    %
    % Inputs:
    % X - Training data (N x d matrix)
    % y - Labels (N x 1 vector)
    % C - Regularization parameter
    % p0 - Initial penalty parameter
    % p_max - Maximum penalty
    % mu0 - Initial Lagrange multiplier estimate
    % beta - Penalty increase multiplier
    % tol - Convergence tolerance
    % max_iter - Maximum iterations
    %
    % Output:
    % lambda_opt - Optimized dual variables
    [N, d] = size(X);
    lambda = zeros(N, 1);  % Initialize lambda
    p = p0;
    mu = mu0;
    Q = (diag(y) * X) * (X' * diag(y));  % Compute the Q matrix
    
    for k = 1:max_iter
        display(k);
        % Solve inner problem using Projected Newton
        lambda_prev = lambda;
        lambda = ProjectedNewton(lambda, Q, y, C, p, mu, tol);
        
        % Update Lagrange multiplier
        g_lambda = sum(lambda .* y);
        mu = 2 * p * g_lambda + mu;
        
        % Update penalty parameter
        p = min(beta * p, p_max);
        
        % Convergence check
        if norm(lambda - lambda_prev) < tol
            fprintf("inner convergence");
            break;
        end
    end
    
    lambda_opt = lambda;

    % Compute optimal w
    w = sum((lambda_opt .* y) .* X, 1)';
    
    % Identify support vectors (0 < lambda < C)
    support_indices = find(lambda_opt > 0 & lambda_opt < C);
    
    % Compute optimal bias w0
    if ~isempty(support_indices)
        w0 = mean(y(support_indices) - X(support_indices, :) * w);
    else
        w0 = 0;
    end
end

function lambda_opt = ProjectedNewton(lambda, Q, y, C, p, mu, tol)
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
    %
    % Output:
    % lambda_opt - Optimized lambda

    epsilon = 1e-6;  % Small regularization term
    H = Q + 2 * p * (y * y') + epsilon * eye(length(lambda));
    grad_F = @(lambda) Q * lambda - ones(size(lambda)) + (2 * p * sum(lambda .* y) + mu) * y;
    
    max_inner_iter = 100;
    for k = 1:max_inner_iter
        % Compute active set: Indices where lambda is at 0 or C
        active_set = (lambda <= 0 & grad_F(lambda) > 0) | (lambda >= C & grad_F(lambda) < 0);
        free_set = ~active_set;
        
        % Compute reduced Hessian
        H_R = H(free_set, free_set);
        grad = grad_F(lambda);
        grad_R = grad(free_set);
        
        % Compute Newton step for free variables
        d_R = -H_R \ grad_R;  % Solve linear system
        d = grad;
        d(free_set) = d_R;
        
        % Perform Armijo line search
        alpha = ArmijoRule(@(l) F_p_mu(l, Q, y, p, mu), lambda, F_p_mu(lambda, Q, y, p, mu), grad_F(lambda), d, 0.2, 0.8, 1e-2, "projNewton");
        
        % Update and project onto feasible region
        lambda_prev = lambda;
        lambda = max(0, min(C, lambda + alpha * d));
        
        % Convergence check
        if norm(lambda - lambda_prev) < tol
            fprintf("outer convergence");
            break;
        end
    end
    
    lambda_opt = lambda;
end

function F = F_p_mu(lambda, Q, y, p, mu)
    % Compute the augmented Lagrangian objective function
    g_lambda = sum(lambda .* y);
    F = 0.5 * lambda' * Q * lambda - sum(lambda) + p * g_lambda^2 + mu * g_lambda;
end