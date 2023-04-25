function [vf, pf, tau, r, w] = pension(pa, a, y, theta, k)
%iterate to solve the tau
maxit = 1e4;
tol = 1e-4;

% guess tau, 
tau = 0.05;



dif = 10;
for it = 1: maxit
    if dif < tol
        break
    end

    % solve the equilibrium r , w
    [vf, pf, r, w] = price(pa, a, y, theta, k, tau);

end


end

