function alpha = ArmijoRule(f, xk, fxk, grad_fxk, dk, sigma, beta, alpha0, Flag)
    % Armijo Rule for Backtracking Line Search
    %
    % Inputs:
    % f       - Function handle to the objective function
    % xk      - Current iterate (point)
    % fxk     - Function value at xk (f(xk))
    % grad_fxk - Gradient of f at xk
    % dk      - Descent direction
    % sigma   - Parameter for Armijo condition (0 < sigma < 1)
    % beta    - Backtracking parameter (0 < beta < 1)
    % alpha0  - Initial step size
    % Flag    - 'projGrad' for projected gradient, 'projNewton' for projected Newton

    alpha = alpha0; % Initialize step size
    x_alpha = computeStep(xk, dk, alpha, Flag);
    fx_alpha = f(x_alpha);

    % Armijo Condition: f(x(alpha)) <= f(xk) + sigma * alpha * grad_fxk' * (x(alpha) - xk)
    while fx_alpha > fxk + sigma * alpha * grad_fxk' * (x_alpha - xk)
        alpha = beta * alpha; % Reduce step size
        x_alpha = computeStep(xk, dk, alpha, Flag);
        fx_alpha = f(x_alpha);
    end
end

function x_alpha = computeStep(xk, dk, alpha, Flag)
    if strcmp(Flag, 'projGrad')
        % Straight line search (unconstrained case)
        x_alpha = xk + alpha * dk;
    elseif strcmp(Flag, 'projNewton')
        % Projection arc search (constrained case)
        a = 0;  % Lower bound of constraint (replace with your problem's constraint)
        b = 0.07;  % Upper bound of constraint (replace with your problem's constraint)
        x_alpha = min(max(xk + alpha * dk, a), b); % Projection on [a, b]
    else
        error('Unknown Flag option');
    end
end