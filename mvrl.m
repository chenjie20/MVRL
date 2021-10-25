function [W, Zn] = mvrl(data_views, alpha)
% ---------------------------------------------------------------------------------------------
%---data_views: nv * m * n
%---nv represents the number of views
%----m and n represent the dimensionality and the number of the features, respectively.
%---------------------------------------------------------------------------------------------

    nv = length(data_views);
    n = size(data_views{1}, 2);
    Xn = cell(1, nv);
    Zn = cell(1, nv);
    Wn = cell(1, nv);

    for idx = 1 : nv        
        Xn{idx} = normc(data_views{idx});  
        XtX =  Xn{idx}' *  Xn{idx};
        Zn{idx} = (XtX + alpha * eye(n)) \ XtX;
                
        [U, s, ~] = svd(Zn{idx}, 'econ');
        s = diag(s);
        r = sum(s>1e-6);

        U = U(:, 1 : r);
        s = diag(s(1 : r));

        M = U * s.^(1/2);
        mm = normr(M);
        rs = mm * mm';
        Wn{idx} = rs.^2;       
    end
    W_sum = zeros(n, n);
    for idx = 1 : nv
        W_sum = W_sum + Wn{idx};
    end    
    W_sum = normc(W_sum);
    W_sum = project_simplex(W_sum); 
    W = (abs(W_sum) + abs(W_sum')) / 2;   

end
