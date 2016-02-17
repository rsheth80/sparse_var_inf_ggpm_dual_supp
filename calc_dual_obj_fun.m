function [dualobj,calcs] = calc_dual_obj_fun(model,data,params)
% function [dualobj,calcs] = calc_dual_obj_fun(model,data,params)
%
% calculates objective function in eq. 24 of "Fast Dual Var Inf for 
% Non-Conjugate LGMs", Khan et al. 2013
% note: the dual optimization is a minimization problem

% dual term arising from conjugate likelihood
dualobj_lik = calc_dualconj(0,model,data,params);
if(isinf(dualobj_lik))
    calcs = [];
    dualobj = dualobj_lik;
    return;
end;

[Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params);
try
    L = chol(Kmm,'lower');
catch
    error('Prior cov is not positive definite');
end;
M = model.Ninducing;
Ntrain = size(data.yt,1);
I = sparse(eye(M));
l = params.dual.lambda;
iKmm = L'\(L\I);

% broke the code up between sod/sparse to save (slightly) on sod comp time
if(~strcmpi(model.type(1:3),'spa'))
    Kmm_tilde = Kmm;
    diagB = zeros(Ntrain,1);
    mu_tilde = fmean_pseudo;
    Alambda = iKmm + diag(l);
else
    C18 = L'\(L\Knm'); % inv(Kmm)*Kmn = W'
    Kmm_tilde = C18'*Kmm*C18; % W*Kmm*W'
    b = fmean_train - C18'*fmean_pseudo; % fmean_train - W*fmean_inducing
    diagB = Kii - diag(Kmm_tilde);  % diag(Knn - W*Kmm*W')
    mu_tilde = C18'*fmean_pseudo + b;
    Alambda = iKmm + C18*diag(l)*C18';
end;

lmy = l - data.yt;
Achol = chol(Alambda,'lower');
Klmy = Kmm_tilde*lmy;

% dual "kld" term (includes terms not dependent on mu, K, or lambda)
dualobj_kld = -sum(log(diag(Achol))) + 0.5*lmy'*Klmy - mu_tilde'*lmy ...
    - 0.5*l'*diagB - sum(log(diag(L)));

% full dual obj (includes terms not dependent on mu, K, or lambda)
dualobj = dualobj_lik + dualobj_kld;

% save some calcs for df
calcs.Achol = Achol;
calcs.diagB = diagB;
calcs.Klmy = Klmy;
calcs.mu_tilde = mu_tilde;
if(~strcmpi(model.type(1:3),'spa'))
else
    calcs.C18 = C18;
end;
