function [vf, pf] = VFI(pa, a, y, theta, k, kh, r, w, tau);
%VFI 
% Input 
% r risk-free rate
% w wage = 1
% tau tax rate for labor's wage
% kh.y sa*sy*stheta -- borrowing constraint for young entrepreneur
% kh.o sa*stheta -- borrowing constraint for old entrepreneur
% Output
% vf.Ve sa*sy*stheta -- young entrep's value function
% vf.Vw sa*sy*stheta -- young worker's vf
% vf.We sa*stheta -- old entrep's vf
% vf.Wr sa -- retiree's vf

% Guess the value functions
vf.Ve = ones(pa.sa, pa.sy, pa.stheta); % young entrepreneur
vf.Vw = ones(pa.sa, pa.sy, pa.stheta); % young worker
vf.We  = ones(pa.sa, pa.stheta); % old entrepreneur
vf.Wr = ones(pa.sa); % retiree

dif = 10;
maxit = 1e4;
tol = 1e-6;

for it = 1: maxit 
    if dif < tol
        break
    end
    % occupation choice
    vf.V = max(vf.Ve, vf.Vw); % young's problem
    vf.W = max(vf.We, repmat(vf.Wr, [1, pa.stheta]) ); % old's problem

    


end


end

