function model = train_cv_svm(data, label_vector)

% Builds and returns the SVM model classifier given the ``feature`` matrix
% of N rows and K columns. Also requires a ``label_vector`` with K entries
% that designates which class the observations belong to.


    % Specify the path to the SVM code (path is for the ``windows`` directory)
    addpath('C:\Program Files\MATLAB\R2008a\work\libsvm-3.1\windows')

    % Try a linear SVM model first ("-t 0" implies a linear model)
    model_linear = svmtrain(label_vector, data, '-t 0');
    % Now a linear model with 5-fold cross-validation: "-v 5"
    model_linear_cv = svmtrain(label_vector, data, '-t 0 -v 5');
    
    % If the linear model gives over 95% accuracy, stop right now and use that
    if model_linear_cv > 95
        model = [];
        model.model = model_linear;
        model.accuracy = model_linear_cv;
        model = append_to_model(model, data, label_vector);
        model.cv_results = NaN;
        return
    end
    
    % See if we can improve the accuracy using an RBF instead (the default
    % SVM model: "-t 2"). Vary "c" = cost parameter = epsilon and "g" = width
    % of the Gaussian kernel.
    log2c_range = -15:2:15;
    log2g_range = -15:2:15;
    bestcv = 0;
    cv_results = meshgrid(log2g_range, log2c_range) * NaN;
    for c_idx = 1:numel(log2c_range)
        for g_idx = 1:numel(log2g_range)
            cmd = ['-q  -t 2  -v 5  -c ', num2str(2^log2c_range(c_idx)), '  -g ', num2str(2^log2g_range(g_idx))];
            cv = svmtrain(label_vector, data, cmd);
            cv_results(c_idx, g_idx) = cv;
            if (cv >= bestcv),
                bestcv = cv; bestc = 2^log2c_range(c_idx); bestg = 2^log2g_range(g_idx);
            end
            fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', c_idx, g_idx, cv, bestc, bestg, bestcv);
        end
    end

    % Pick the settings of (c, g) that give the highest cross-validation accuracy
    % and rebuild the model at those settings.
    cmd = ['-t 2  -c ', num2str(bestc), '  -g ', num2str(bestg)];
    model_rbf = svmtrain(label_vector, data, cmd);

    % Test the predictions on the training set
    % ``accuracy`` = [accuracy, mean squared error, squared correlation]
    [predict_label, accuracy] = svmpredict(label_vector, data, model_rbf);
       
    % Understand the model
    model = [];
    model.model = model_rbf;
    model.accuracy = accuracy;
    model.cv_results = cv_results;
end
