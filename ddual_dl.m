function df = ddual_dl(model,data,params,calcs)

% dual obj gradient due to conjugate likelihood
df_lik = calc_dualconj(1,model,data,params);
if(any(isinf(df_lik)))
    df = df_lik;
    return;
end;

if(nargin==4&&~isempty(calcs))
    iscalcs = 1;
else
    iscalcs = 0;
end;

M = model.Ninducing;
I = sparse(eye(M));

if(~iscalcs)
    l = params.dual.lambda;
    [Kmm,Knm,Kii,fmean_pseudo,fmean_train] = compute_model(model,data,params);
    L = chol(Kmm,'lower'); % should have checked for non-sing Kmm before this point
    iKmm = L'\(L\I);
    if(~strcmpi(model.type(1:3),'spa'))
        Kmm_tilde = Kmm;
        diagB = zeros(size(data.yt,1),1);
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
else
    Achol = calcs.Achol;
    diagB = calcs.diagB;
    Klmy = calcs.Klmy;
    mu_tilde = calcs.mu_tilde;
    if(~strcmpi(model.type(1:3),'spa'))
    else
        C18 = calcs.C18;
    end;
end;

if(~strcmpi(model.type(1:3),'spa')) % sod
    x = Achol\I;
else
    x = Achol\C18;
end;

% dual obj gradient due to "kld"
df_kld = Klmy - mu_tilde - 0.5*(sum(x.^2)' + diagB);

% total dual obj gradient
df = df_kld + df_lik;
