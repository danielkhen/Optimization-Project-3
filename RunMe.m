C = 1;
p0 = 1;
p_max = 1000;
mu0 = 1;
p_mult = 2;
tol = 1e-4;
max_iter = 100;
sigma = 0.2;
beta = 0.8;
alpha0 = 1e-2;
Flag = input("Insert flag projected gradient descent (Flag = ’projGrad’) and projected Newton (Flag = ’projNewton’):", "s");

if Flag ~= "projGrad" && Flag ~= "projNewton"
    error("flag must be projGrad or projNewton");
end

X = load("xForTraining.mat").xForTraining;
y = load("labelsForTraining.mat").labelsForTraining;
y(y==0) = -1;
y(y==9) = 1;
coeff = load("coeff.mat").coeff;
X = ExtractFeatures(X, coeff, 50)';
[lambda, w, w0] = AugmentedLagrangianSVM(X, y, C, p0, p_max, mu0, beta, tol, max_iter, Flag, sigma, beta, alpha0);

X_test = load("xForTest.mat").xForTest;
y_test = load("labelsForTest.mat").labelsForTest;
y_test(y_test==0) = -1;
y_test(y_test==9) = 1;
X_test = ExtractFeatures(X_test, coeff, 50)';
results = X_test * w + w0;
results(results >= 0) = 1;
results(results < 0) = -1;
test_size = size(y_test, 1);
accuracy = sum(results == y_test) * 100 / test_size;
misclassifications = sum(results ~= y_test);
fprintf("Accuracy: %.4g\n", accuracy);
fprintf("%i out of %i images were misclassified", misclassifications, test_size);