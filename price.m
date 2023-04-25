function [vf, pf, r, w] = price(pa, a, y, theta, k, tau)
%PRICE solve for the equilibirium r, and w, given tax rate, tau

% guess r and w 
r = 0.05;
w = 1; % numeraire

maxit = 1e4;
dif = 10;
tol = 1e-5;

for it = 1: maxit 
    if dif < tol 
        break
    end

    % solve the borrowing constraint
    [vf, pf, kh] = bor_con(pa, a, y, theta, k, r, w, tau);
    




end


end

