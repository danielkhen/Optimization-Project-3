function alpha = ArmijoRule(f, xk, fxk, grad_fxk, dk, sigma, beta, alpha0, Flag, C)
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
    % Flag    - 'boxConstraints' for box constraints projection to [0, C],
    % 'noConstraints' for no projection
    % C       - higher bound of box constraint

    alpha = alpha0; % Initialize step size
    x_alpha = computeStep(xk, dk, alpha, Flag, C);
    fx_alpha = f(x_alpha);

    % Armijo Condition
    while fx_alpha - fxk > sigma * grad_fxk' * (x_alpha - xk)
        alpha = beta * alpha; % Reduce step size
        x_alpha = computeStep(xk, dk, alpha, Flag, C);
        fx_alpha = f(x_alpha);
    end
end

function x_alpha = computeStep(xk, dk, alpha, Flag, C)
    % Should project to box?
    if strcmp(Flag, 'boxConstraints')
        x_alpha = min(max(xk + alpha * dk, 0), C); % Projection on [0, C]
    else
        x_alpha = xk + alpha * dk;
    end
end