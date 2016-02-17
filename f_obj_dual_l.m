function [f,df] = f_obj_dual_l(x,model,data,params,dummy_arg)
% function [f,df] = f_obj_dual_l(x,model,data,params)
% function lambda = f_obj_dual_l(params)
% function params = f_obj_dual_l(x,model,data,params,dummy_arg)
%
% note: the dual optimization is a minimization problem

if(nargin==1)
    f = x.dual.lambda;
    df = [];
    return;
end;

if(nargin==5)
    [Kmm,Knm,~,fmean_pseudo] = compute_model(model,data,params);
    L = chol(Kmm,'lower');
    params_update = params;
    params_update.dual.lambda = x;
    alpha = x - data.yt;
    I = eye(model.Ninducing);
    if(~strcmpi(model.type(1:3),'spa'))
        C18 = I;
    else
        C18 = L'\(L\Knm'); % inv(Kmm)*Kmn = W'
    end;
    params_update.var.m = fmean_pseudo - Kmm*C18*alpha;
    Alambda = L'\(L\I) + C18*diag(x)*C18';
    Achol = chol(Alambda);
    params_update.var.C = Achol\I;
    f = params_update;
    df = [];
    return;
end;

params_update = params;
params_update.dual.lambda = x;
[f,calcs] = calc_dual_obj_fun(model,data,params_update);
df = ddual_dl(model,data,params_update,calcs);
