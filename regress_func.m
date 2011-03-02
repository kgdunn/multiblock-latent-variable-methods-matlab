function b = regress_func(Y, x, has_missing)
% Regress vector ``x`` onto the columns in matrix ``Y`` one at a time.
% Return the vector of regression coefficients, one for each column in ``Y``.
% There may be missing data in ``Y``, but never in ``x``.  
% The ``x`` vector *must* be a column vector.

% Supply the true/false indicator, ``has_missing`` to avoid slower calculations
% for missing data.


[Ny, K] = size(Y);
Nx = numel(x);

if Ny == Nx                  % Case A: b' = (x'Y)/(x'x): (1xN)(NxK) = (1xK)
    if not(has_missing)
        b = (Y'*x)/(x'*x);
        return
    end
    b = zeros(K, 1);
    for k = 1:K
        keep = ~isnan(Y(:,k));
        b(k) = sum(x(keep, 1) .* Y(keep, k));
        denom = norm(x(keep))^2;
        if abs(denom) > eps
            b(k) = b(k) / denom;
        end
    end
elseif K == Nx
    % Case B:  b = (Yx)/(x'x): (NxK)(Kx1) = (Nx1)
    % Regressing x onto rows in Y, storing results in column vector "b"
    if not(has_missing)
        b = (Y*x)/(x'*x);
        return
    end
        
    b = zeros(Ny, 1);    
    for n = 1:Ny
        keep = ~isnan(Y(n, :));
        b(n) = sum(x(keep)' .* Y(n, keep)); %    sum(x(:,0) * np.nan_to_num(Y(n,:)));
        % TODO(KGD): check: this denom is usually(always?) equal to 1.0
        denom = norm(x(keep))^2;
        if abs(denom) > eps
            b(n) = b(n) / denom;
        end

    end
end
