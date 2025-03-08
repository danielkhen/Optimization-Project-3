X = load("xForTraining.mat").xForTraining;
y = load("labelsForTraining.mat").labelsForTraining;
y(y==0) = -1;
y(y==9) = 1;
coeff = load("coeff.mat").coeff;
X = ExtractFeatures(X, coeff, 50)';
[w, w0] = AugmentedLagrangianSVM(X, y, 0.07, 1, 1000, 1, 2, 1e-4, 100);

X_test = load("xForTest.mat").xForTest;
y_test = load("labelsForTest.mat").labelsForTest;
y_test(y_test==0) = -1;
y_test(y_test==9) = 1;
X_test = ExtractFeatures(X_test, coeff, 50)';
results = X_test * w + w0;
results(results >= 0) = 1;
results(results < 0) = -1;
display(sum(results == y_test) * 100 / size(y_test, 1));