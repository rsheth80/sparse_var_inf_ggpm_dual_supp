function max_step = dual_max_step_func(x,d,t)

e = 1e-12*0;
ix = find(x+d*t<0);
if(isempty(ix))
    m = t;
else
    m = min(x(ix)./abs(d(ix)));
end;
max_step = (1-e)*min(m,t);
