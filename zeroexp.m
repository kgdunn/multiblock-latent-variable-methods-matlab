function out = zeroexp(shape, varargin)

    % Will return an array of zeros of the given ``shape``.  Or if given an
    % existing array ``prev`` will expand it to the new ``shape``, maintaining
    % the existing elements, but adding zeros to the new elements.  Only
    % one of the elements in ``shape`` may be different from size(prev)

    out = zeros(shape);
    if nargin == 3
        % If "true" then expand it with NaN's rather than zeros
        if varargin{2}
            out = out .* NaN;
        end
    end
    if nargin == 2 && not(isempty(varargin{1}))
        prev = varargin{1};
        prev_shape = size(prev);
    else
        return;
    end
    

    % Which dimension is to be expanded?
    which_dim = find(shape-prev_shape);
    if numel(which_dim) > 1
        error('The new shape must have all dimensions, except one, as the same size.');
    end
    % If the dimensions are the same as required, just pass the input as
    % the output
    if isempty(which_dim)
        out = prev;
    else
        idx = cell(1,numel(shape));
        for k = 1:numel(shape)
            idx{k} = ':';
        end
        idx{which_dim} = 1:prev_shape(which_dim);        
        subsarr = struct('type', '()', 'subs', {idx});
        subsasgn(out, subsarr, prev);
        
    end
    
end
