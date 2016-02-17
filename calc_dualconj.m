function x = calc_dualconj(order,model,data,params)
% function lambda0 = calc_dualconj(-1,model,data,params)
% function dualobj_lik = calc_dualconj(0,model,data,params)
% function ddualobj_dl_lik = calc_dualconj(1,model,data,params)

if(order>=0)
    l = params.dual.lambda;
end;
model_ltype = strrep(strrep(model.type,'sod_',''),'sparse_','');

switch(order)
case -1
    if(strcmp(model.type(1:2),'so')) % sod model
        [Kmm,~,~,fmean_pseudo] = compute_model(model,data,params);
        l0 = data.yt + Kmm\(fmean_pseudo - params.var.m);
        switch(lower(model_ltype))
        case 'gppr'
            l0(l0<=0) = 1e-3;
        case 'gpc'
            l0(l0<=0) = 1e-3;
            l0(l0>=1) = 1-1e-3;
        end;
    else % otherwise, it's anyone's guess how to initialize lambda
        switch(lower(model_ltype))
        case 'gppr'
            l0 = ones(size(data.yt));
        case 'gpc'
            l0 = 0.5*ones(size(data.yt));
            %l0 = 0.75*ones(size(data.yt));
            %l0(data.yt==-1) = 0.25;
        case 'gpo'
            l0 = zeros(size(data.yt)); % ?
        end;
    end;
    x = l0;
case 0
    switch(lower(model_ltype))
    case 'gppr'
        if(any(l<=0))
            dual_obj_lik = inf;
        else
            dual_obj_lik = l'*(log(l)-1) - sum(gammaln(data.yt+1));
        end;
    case 'gpc'
        if(any(l<=0)||any(l>=1))
            dual_obj_lik = inf;
        else
            dual_obj_lik = sum(l.*log(l)+(1-l).*log(1-l));
        end;
    end;
    x = dual_obj_lik;
case 1
    switch(lower(model_ltype))
    case 'gppr'
        if(any(l<=0))
            ddualobj_dl_lik = inf(size(l));
        else
            ddualobj_dl_lik = log(l);
        end;
    case 'gpc'
        if(any(l<=0)||any(l>=1))
            ddualobj_dl_lik = inf(size(l));
        else
            ddualobj_dl_lik = log(l) - log(1-l);
        end;
    end;
    x = ddualobj_dl_lik;
end;
