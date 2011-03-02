function out = ssq(X, axis)

% A function than calculates the sum of squares of a matrix (not array!),
% skipping over any NaN (missing) data.
%
% If ``axis`` is not specified, it will sum over the entire array and
% return a scalar value.  If ``axis`` is specified, then the output is
% usually a vector, with the sum of squares taken along that axis.
%
% If a complete dimension has missing values, then ssq will return 0.0 for
% that sum of squares.
%
% Relies on nansum.m


if nargin == 1
    out = nansum(X(:).*X(:));
else
    out = nansum(X.*X, axis);
end
out(isnan(out)) = 0.0;