function [vf, pf, kh] = bor_con(pa, a, y, theta, k, r, w, tau)
%BOR_CON sovle the borrowing constraint, kh(x)
% input 
% r risk-free rate
% w wage = 1
% tau tax rate for labor's wage

% output
% kh.y sa*sy*stheta -- borrowing constraint for young entrepreneur
% kh.o sa*stheta -- borrowing constraint for old entrepreneur

dif = 10; 
tol = 1e-5;
maxit = 1e4;
% guess kh
kh.y = pa.maxk * ones(pa.sa, pa.sy, pa.stheta);
kh.o = pa.maxk * ones(pa.sa, pa.stheta);

for it = 1: maxit
    if dif < tol
        break
    end

    % value function iteration with borrowing constraint
    [vf, pf] = VFI(pa, a, y, theta, k, kh, r, w, tau);
    

    % update the kh
    

end



end


