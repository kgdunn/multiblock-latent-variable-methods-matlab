function test_blocks(varargin)
    close all;
    
    test_basic_syntax();
    test_exclude();
    
    
    test_batch_blocks();
    test_labels();
    test_preprocessing();
    test_plotting();
    disp('ALL TESTS FOR THE BLOCK OBJECT PASSED - if no errors were displayed.')
end

function test_basic_syntax()    
    % 1-D and 2-D tests
    b = block(NaN);
    assertTrue(b.N == 1)
    assertTrue(b.K == 1)
    assertTrue(strcmpi(b.name, 'block-1-1'))
    assertTrue(b.has_missing == true);
    
    
    X = randn(10, 5);
    b = block(X);
    assertTrue(b.N == 10)
    assertTrue(b.K == 5)
    assertTrue(all(b.data(:) == X(:)))
    assertTrue(b.has_missing == false);
    assertTrue(strcmpi(b.name_type, 'auto'))
    assertTrue(strcmpi(b.name, 'block-10-5'))
    
    assertTrue(all(shape(b) == [10, 5]))
    assertTrue(shape(b, 1) == 10)
    assertTrue(shape(b, 2) == 5)
    
    
    b = block([]);
    assertTrue(b.N == 0)
    assertTrue(b.K == 0)
    assertTrue(strcmpi(b.name, 'block-0-0'))
    assertTrue(isempty(b))
    
    b = block(24);
    assertTrue(b.N == 1)
    assertTrue(b.K == 1)
    assertTrue(strcmpi(b.name, 'block-1-1'))
    assertTrue(b.has_missing == false);
    
    
end

function test_exclude()
% Excluding rows from the block

    % Array blocks
    FMC = load('datasets/FMC.mat');
    Z = block(FMC.Z);
    Z.add_labels(2, FMC.Znames)
    
    [Z, other] = Z.exclude(1, 12:15)
    assertTrue(all(shape(Z) == [55, 20]))
    
    [Z, other] = Z.exclude(2, [9:14 16])
    assertTrue(all(shape(Z) == [55, 13]))

    % Batch blocks
    
%     batch_names = {'A', 'B', 'C', 'Four', 'E', 'F', 'G', 'H', 'I', 'Ten'};
%     time_names = {'1', '2', '3', '4', '5'};
%     tag_names = {'Temp', 'Pres', 'Flow', 'Heat input', 'Flow2'};
%     b = block(X, 'X block', {'batch_names', batch_names}, ...
%                             {'batch_tag_names', tag_names}, ....
%                             {'time_names', time_names});
%                         
%     b.exclude(1, 4);
%     b.exclude(1, 'A');
%     
%     b.exclude(1, {'A'});
%     b.exclude(1, {'A', 'H'});
end

function test_batch_blocks()
% Batch data
% -----------
% block(data, block name, {'nBatches', num batches})
% block(data, block name, {'batch_tag_names', tag_names}, ... <-- name of each tag in batch  
%                         {'time_names', time_names}, ...     <-- cell array, or vector of integers
%                         {'batch_names', batch_names})       <-- name of each batch

    X = randn(120, 5);
    
    b = block(X, 'batch block', {'nBatches', 12});
    assertTrue(b.N == 12)
    assertTrue(b.K == 50)
    assertTrue(b.nTags == 5)
    assertTrue(b.J == 10)
    assertTrue(strcmpi(class(b), 'block_batch'))
    assertTrue(strcmpi(b.name, 'batch block'))
    assertTrue(strcmpi(b.name_type, 'given'))
    assertTrue(all(shape(b) == [12, 5, 10]))
    assertTrue(shape(b, 1) == 12)
    assertTrue(shape(b, 2) == 5)
    assertTrue(shape(b, 3) == 10)
    
    
    tag_names = {'A', 'B', 'C', 'D', 'E'};
    b = block(X, 'batch block', {'nBatches', 12}, {'batch_tag_names', tag_names});
    assertTrue(b.N == 12)
    assertTrue(b.K == 50)
    assertTrue(b.nTags == 5)
    assertTrue(b.J == 10)
    assertTrue(all(all(char(b.labels{2,1}) == char(tag_names))))
    
    time_names = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    b = block(X, 'batch block', {'nBatches', 12}, {'time_names', time_names});
    assertTrue(b.N == 12)
    assertTrue(b.K == 50)
    assertTrue(b.nTags == 5)
    assertTrue(b.J == 10)
    internal_rep = cellstr(num2str(cell2mat(time_names(:))));
    assertTrue(all(all(char(b.labels{3}) == char(internal_rep))))
    

    batch_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'};
    b = block(X, {'batch_names', batch_names});
    assertTrue(b.N == 12)
    assertTrue(b.K == 50)
    assertTrue(b.nTags == 5)
    assertTrue(b.J == 10)
    assertTrue(strcmp(b.name, 'batch-12-50-10'));
    assertTrue(all(cell2mat(b.labels{1}) == cell2mat(batch_names)))
    
    % Check the batch data structure
    assertTrue(all(size(b.data) == [12, 50]))
    assertTrue(all(size(b.batch_raw) == [12, 1]))
    assertTrue(all(size(b.batch_raw{1}) == [10, 5]))
    assertTrue(all(all(b.batch_raw{1} == X(1:10,:))))    
    
    % Access to indicies via batch labels
    X = randn(50,4);
    batch_names = {'A', 'B', 'C', 'Four', 'E', 'F', 'G', 'H', 'I', 'Ten'};
    time_names = {'1', '2', '3', '4', '5'};
    tag_names = {'Temp', 'Pres', 'Flow', 'Heat input', 'Flow2'};
    b = block(X, 'X block', {'batch_names', batch_names}, ...
                            {'batch_tag_names', tag_names}, ....
                            {'time_names', time_names});
                        
                        
    [mark, mark_names] = b.index_by_name(1, 'Four');
    assertTrue(mark == 4)
    assertTrue(strcmp(mark_names{1}, 'Four'))
    
    [mark, mark_names] = b.index_by_name(1, {'Four', 'E'});
    assertTrue(all(mark == [4, 5]))
    assertTrue(strcmp(mark_names{1}, 'Four'))
    assertTrue(strcmp(mark_names{2}, 'E'))
    
    % Non-existant names 
    [mark, mark_names] = b.index_by_name(1, ['Four', 'E']);
    assertTrue(all(mark == []))
    assertTrue(isempty(mark_names));
    
    % Non-existant names 
    [mark, mark_names] = b.index_by_name(1, {'Four', 'Not here', 'Ten'});
    assertTrue(all(mark == [4, 10]))
    assertTrue(strcmp(mark_names{1}, 'Four'))
    assertTrue(strcmp(mark_names{2}, 'Ten'))
    
    % Other dimensions
    [mark, mark_names] = b.index_by_name(2, {'Temp', 'Flow'});
    assertTrue(all(mark == [1, 3]))
    assertTrue(strcmp(mark_names{1}, 'Temp'))
    assertTrue(strcmp(mark_names{2}, 'Flow'))
    
    % Index by indexing: currently does not work (TODO)
    [mark, mark_names] = b.index_by_name(1, [7, 4]);
    assertTrue(all(mark == []))
    assertTrue(isempty(mark_names));
end

function test_labels()
    X = randn(10,4);
    row_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'Ten'};
    tag_names = {'Temp', 'Pres', 'Flow', 'Heat input'};
    b = block(X, 'X block', {'row_labels', row_names}, ...
                            {'col_labels', tag_names});

    assertTrue(all(all(char(b.labels{1,1}) == char(row_names))))    
    assertTrue(all(all(char(b.labels{2,1}) == char(tag_names))))
    
    b.add_labels(1, row_names);
    assertTrue(all(all(char(b.labels{1,2}) == char(row_names))))    
    
    
    X = randn(50,4);
    batch_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'Ten'};
    time_names = {'1', '2', '3', '4', '5'};
    tag_names = {'Temp', 'Pres', 'Flow', 'Heat input', 'Flow2'};
    b = block(X, 'X block', {'batch_names', batch_names}, ...
                            {'batch_tag_names', tag_names}, ....
                            {'time_names', time_names});
    assertTrue(all(all(char(b.labels{1,1}) == char(batch_names))))
    assertTrue(all(all(char(b.labels{2,1}) == char(tag_names))))
    assertTrue(all(all(char(b.labels{3,1}) == char(time_names))))

end

function test_plotting()

    % Plot an array with 4 columns
    X = randn(10,4);
    row_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'Ten'};
    tag_names = {'Temp', 'Pres', 'Flow', 'Heat input'};
    b = block(X, 'X block', {'row_labels', row_names}, ...
                            {'col_labels', tag_names});
    close all;
    h = plot(b);
    assert(numel(h) == 4);
    f = gcf;
    assert(get(h(1), 'Parent') == f);
    delete(f)
    
    % Plot an array with 5 columns
    X = randn(10,5);
    row_names = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'Ten'};
    tag_names = {'Temp', 'Pres', 'Flow', 'Heat input', 'Test'};
    b = block(X, 'X block', {'row_labels', row_names}, ...
                            {'col_labels', tag_names});
    h = plot(b);
    assert(numel(h) == 5);
    f = gcf;
    assert(get(h(1), 'Parent') == f);
    delete(f)
    
    % Plot a batch block
    X = randn(50,4);
    batch_names = {'A', 'B', 'C', 'Four', 'E', 'F', 'G', 'H', 'I', 'Ten'};
    time_names = {'1', '2', '3', '4', '5'};
    tag_names = {'Temp', 'Pres', 'Flow', 'Heat input', 'Flow2'};
    b = block(X, 'X block', {'batch_names', batch_names}, ...
                            {'batch_tag_names', tag_names}, ....
                            {'time_names', time_names});
                        
     
                        
    % Basic batch block plot: shows each trajectory for all batches
    h = plot(b);
    assert(numel(h) == 4);
    f = gcf;
    assert(get(h(1), 'Parent') == f);
    delete(f)
    
    % Specify the layout for the trajectories: 1x2 subplots per window
    h = plot(b, {'layout', [1, 2]});
    assert(numel(h) == 4);
    hF = unique(cell2mat(get(h, 'Parent')));
    assert(numel(hF) == 2);
    for f = 1:numel(hF)
        delete(hF(f))
    end
    
    
    % Specify a batch to highlight
    h = plot(b, {'mark', 'Four'});
    assert(numel(h) == 4);
    f = gcf;
    assert(get(h(1), 'Parent') == f);
    delete(f)
    
    % Specify batches to highlight
    h = plot(b, {'mark', {'Four', 'E'}});
    assert(numel(h) == 4);
    f = gcf;
    assert(get(h(1), 'Parent') == f);
    delete(f)
    
    % Show the preprocessed batch data instead (the default is to show the raw
    % data).  TODO
    %h = plot(b, {'pp', true});
    
    
    % Load an actual batch data set
    dupont = load('datasets/DuPont.mat');
    batch_data = dupont.tech;

    % Specify the data dimensions
    nBatches = 55;
    tagNames = {'TempR-1','TempR-2','TempR-3','Press-1', ...
                'Flow-1', 'TempH-1','TempC-1','Press-2', ...
                'Press-3','Flow-2'};

    % We must create a batch block first: tell it how
    % many batches there are in the aligned data
    b = block(batch_data, 'DuPont X', {'batch_tag_names', tagNames}, ...
                           {'nBatches', nBatches});
    h = plot(b, {'layout', [2,2]}, {'mark', 32});
    assert(numel(h) == 10);
    hF = unique(cell2mat(get(h, 'Parent')));
    assert(numel(hF) == 3);
    for f = 1:numel(hF)
        delete(hF(f))
    end
    
end

function test_preprocessing()

    % No missing data
    LDPE = load('tests/LDPE-PCA.mat');
    raw_data = LDPE.data;  % Raw data
    
    % Test that mean centering and scaling are correct
    X = block(raw_data.blocks{1});
    X = X.preprocess();
    assertElementsAlmostEqual(X.data, raw_data.scaled_blocks{1}, 4)
    
    % Missing data
    X = block([1 2 3; 3 NaN 5; 5, 7, NaN]);
    [out, PP] = X.preprocess();
    assertElementsAlmostEqual(PP.mean_center, [3.0, 4.5,  4.0], 4)
    assertElementsAlmostEqual(PP.scaling, [0.5, 0.282828282,  1/sqrt(2)], 4)
    
    % Raw data is untouched
    assertElementsAlmostEqual(X.data, [1 2 3; 3 NaN 5; 5, 7, NaN], 4)
    
    % Output is preprocessed
    assertElementsAlmostEqual(out.data, [-1 -1/sqrt(2) -1/sqrt(2); 0 NaN 1/sqrt(2); 1 1/sqrt(2) NaN], 4)
    
    
    X_raw = [3, 4, 2, 2; 4, 3, 4, 3; 5.0, 5, 6, 4];
    X = block(X_raw);
    [data, PP] = preprocess(X);  % or X = X.preprocess();

    % The mean centering vector should be [4, 4, 4, 3], page 40
    assertTrue(all(PP.mean_center == [4, 4, 4, 3]));

    % The (inverted) scaling vector [1, 1, 0.5, 1], page 40
    assertTrue(all(PP.scaling == [1, 1, 0.5, 1]));
    
    
    % Batch preprocessing
    dupont = load('datasets/DuPont.mat');
    batch_data = dupont.tech;

    % Specify the data dimensions
    nBatches = 55;
    tagNames = {'TempR-1','TempR-2','TempR-3','Press-1', ...
                'Flow-1', 'TempH-1','TempC-1','Press-2', ...
                'Press-3','Flow-2'};

    b = block(batch_data, 'DuPont X', {'batch_tag_names', tagNames}, ...
                           {'nBatches', nBatches});
    [data_PP, PP] = preprocess(b); 

    % The mean centering vector should be [4, 4, 4, 3], page 40
    assertTrue(numel(PP.mean_center()) == 1000);

    % The (inverted) scaling vector [1, 1, 0.5, 1], page 40
    assertTrue(numel(PP.scaling) == 1000);
    
    
    % Apply preprocessing to a new block
    new_data = block(batch_data(1:200, :), 'DuPont X', ...
            {'batch_tag_names', tagNames},  {'nBatches', 2});
        
    new = b.preprocess(new_data, PP);
    assertElementsAlmostEqual(new.data, data_PP.data(1:2,:), 8)
end


% =========================================================================
% Code from this point onwards is from the MATLAB open-source unit testing 
% suite: http://www.mathworks.com/matlabcentral/fileexchange/22846
%
% Licensed under the BSD license, which means it may be legally
% distributed - see below.  Their code has been modified, where required,
% to make it work with earlier MATLAB versions.
%
%
% Copyright (c) 2010, The MathWorks, Inc.
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the The MathWorks, Inc. nor the names 
%       of its contributors may be used to endorse or promote products derived 
%       from this software without specific prior written permission.
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.
% -----------------

function assertTrue(condition)
% Verifies that the ``condition`` is true.  If not, it will throw an
% exception.

%assertTrue Assert that input condition is true
%   assertTrue(CONDITION, MESSAGE) throws an exception containing the string
%   MESSAGE if CONDITION is not true.
%
%   MESSAGE is optional.
%
%   Examples
%   --------
%   % This call returns silently.
%   assertTrue(rand < 1, 'Expected output of rand to be less than 1')
%
%   % This call throws an error.
%   assertTrue(sum(sum(magic(3))) == 0, ...
%       'Expected sum of elements of magic(3) to be 0')
%
%   See also assertEqual, assertFalse

%   Steven L. Eddins
%   Copyright 2008-2010 The MathWorks, Inc.

    if nargin < 2
       message = 'Asserted condition is not true.';
    end

    if ~isscalar(condition) || ~islogical(condition)
       throwAsCaller(MException('assertTrue:invalidCondition', ...
          'CONDITION must be a scalar logical value.'));
    end

    if ~condition
       throwAsCaller(MException('assertTrue:falseCondition', '%s', message));
    end
end

function assertElementsAlmostEqual(A, B, sigfigs, ignore_sign)
% KGD: original code description below.  In this case, sigfigs = number of
% significant figures.

%assertElementsAlmostEqual Assert floating-point array elements almost equal.
%   assertElementsAlmostEqual(A, B, tol_type, tol, floor_tol) asserts that all
%   elements of floating-point arrays A and B are equal within some tolerance.
%   tol_type can be 'relative' or 'absolute'.  tol and floor_tol are scalar
%   tolerance values.
%
%   If the tolerance type is 'relative', then the tolerance test used is:
%
%       all( abs(A(:) - B(:)) <= tol * max(abs(A(:)), abs(B(:))) + floor_tol )
%
%   If the tolerance type is 'absolute', then the tolerance test used is:
%
%       all( abs(A(:) - B(:)) <= tol )
%
%   tol_type, tol, and floor_tol are all optional.  The default value for
%   tol_type is 'relative'.  If both A and B are double, then the default value
%   for tol and floor_tol is sqrt(eps).  If either A or B is single, then the
%   default value for tol and floor_tol is sqrt(eps('single')).
%
%   If A or B is complex, then the tolerance test is applied independently to
%   the real and imaginary parts.
%
%   Corresponding elements in A and B that are both NaN, or are both infinite
%   with the same sign, are considered to pass the tolerance test.
%
%   assertElementsAlmostEqual(A, B, ..., msg) prepends the string msg to the
%   output message if A and B fail the tolerance test.

%   Steven L. Eddins
%   Copyright 2008-2010 The MathWorks, Inc.

if nargin < 4
    ignore_sign = false;
end

if ~isequal(size(A), size(B))
    message = 'Inputs are not the same size.';
    throwAsCaller(MException('assertElementsAlmostEqual:sizeMismatch', ...
        '%s', message));
end

if ~(isfloat(A) && isfloat(B))
    message = 'Inputs are not both floating-point.';
    throwAsCaller(MException('assertElementsAlmostEqual:notFloat', ...
        '%s', message));
end

if ~isfloat(A) || ~isfloat(B)
    return_early = isequal(A, B);
else
    return_early = false;
end

if ~isequal(size(A), size(B))
    return_early = false;
end

A = A(:);
B = B(:);

reltol = max(100 * max(eps(class(A)), eps(class(B))), 10^(-sigfigs));
if ignore_sign    
    delta = abs(abs(A) - abs(B)) ./ max(max(abs(A), abs(B)), reltol);
else
    delta = abs(A - B) ./ max(max(abs(A), abs(B)), reltol);
end

% Some floating-point values require special handling.
delta((A == 0) & (B == 0)) = 0;
delta(isnan(A) & isnan(B)) = 0;
delta((A == Inf) & (B == Inf)) = 0;
delta((A == -Inf) & (B == -Inf)) = 0;
same = all(delta <= reltol);

if ~same || return_early
    tolerance_message = sprintf('Input elements are not all equal within a relative tolerance of %g', reltol);
    throwAsCaller(MException('assertElementsAlmostEqual:tolExceeded', ...
        '%s', tolerance_message));
end

end

function assertExceptionThrown(expectedId, f, varargin)
% KGD: I have changed the function signature to be compatible with Python.

%assertExceptionThrown Assert that specified exception is thrown
%   assertExceptionThrown(F, expectedId) calls the function handle F with no
%   input arguments.  If the result is a thrown exception whose identifier is
%   expectedId, then assertExceptionThrown returns silently.  If no exception is
%   thrown, then assertExceptionThrown throws an exception with identifier equal
%   to 'assertExceptionThrown:noException'.  If a different exception is thrown,
%   then assertExceptionThrown throws an exception identifier equal to
%   'assertExceptionThrown:wrongException'.
%
%   assertExceptionThrown(F, expectedId, msg) prepends the string msg to the
%   assertion message.
%
%   Example
%   -------
%   % This call returns silently.
%   f = @() error('a:b:c', 'error message');
%   assertExceptionThrown(f, 'a:b:c');
%
%   % This call returns silently.
%   assertExceptionThrown(@() sin, 'MATLAB:minrhs');
%
%   % This call throws an error because calling sin(pi) does not error.
%   assertExceptionThrown(@() sin(pi), 'MATLAB:foo');

%   Steven L. Eddins
%   Copyright 2008-2010 The MathWorks, Inc.

    noException = false;
    try
        f(varargin{:});
        noException = true;

    catch exception
        if ~strcmp(exception.identifier, expectedId)
            message = sprintf('Expected exception %s but got exception %s.', ...
                expectedId, exception.identifier);
            throwAsCaller(MException('assertExceptionThrown:wrongException', ...
                '%s', message));
        end
    end

    if noException
        message = sprintf('Expected exception "%s", but none thrown.', ...
            expectedId);
        throwAsCaller(MException('assertExceptionThrown:noException', '%s', message));
    end
end