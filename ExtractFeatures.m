function [x_] = ExtractFeatures(x,coeff,NumofPrincComp)
%{
Extracts 50 principle components of x
Input - x - vector of raw data
        coeff - basis tranformation matrix
        NumofPrincComp - number of largest principle components

Output - x_ - "NumofPrincComp" lragest principle components of x
%}
x_=coeff.'*x;
x_=x_(1:NumofPrincComp,:);

end

