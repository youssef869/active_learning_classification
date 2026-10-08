clear; clc; close all;

% =========================================================
% PIPELINE 2 — Raw Pixels (no HOG)
%
% Simpler and more stable than HOG for this dataset.
% Raw pixel vectors (784-dim) with StandardScaler inside
% fitcecoc give a well-behaved RBF SVM without any
% feature extraction complications.
% =========================================================

% ---------------------------------------------------------
% STEP 1: Load full dataset → X_all (784-dim pixel vectors)
% ---------------------------------------------------------
folder = 'Indian_Digits_Train';
files  = dir(fullfile(folder, '*.bmp'));

% Sort files numerically (1.bmp, 2.bmp, ... 10000.bmp)
fileNums = zeros(length(files), 1);
for i = 1:length(files)
    fileNums(i) = str2double(erase(files(i).name, '.bmp'));
end
[~, order] = sort(fileNums);
files    = files(order);
fileNums = fileNums(order);

numImages = length(files);
pixDim    = 28 * 28;   % 784

X_all = zeros(numImages, pixDim, 'single');

fprintf('Loading %d images...\n', numImages);
for i = 1:numImages
    img        = imread(fullfile(folder, files(i).name));
    X_all(i,:) = reshape(im2single(img), 1, []);
end
fprintf('Done. X_all: %d x %d\n', size(X_all,1), size(X_all,2));

% Lookup: filename number → row index in X_all
numToIdx = containers.Map(num2cell(fileNums), num2cell(1:numImages));

% ---------------------------------------------------------
% STEP 2a: Load seed labels from CSV
% ---------------------------------------------------------
T       = readtable('seed_labels.csv', 'TextType', 'string');
numSeed = height(T);
fprintf('\nLoaded %d seed labels from CSV.\n', numSeed);

seedPix    = zeros(numSeed, pixDim, 'single');
seedLabels = zeros(numSeed, 1);
seedRowIdx = zeros(numSeed, 1);

% Handle both numeric and string Label columns
labelIsNumeric = isnumeric(T.Label);

for i = 1:numSeed
    imgNum        = sscanf(char(T.Filename(i)), '%d.bmp');
    rowIdx        = numToIdx(imgNum);
    seedPix(i,:)  = X_all(rowIdx,:);
    if labelIsNumeric
        seedLabels(i) = T.Label(i);
    else
        seedLabels(i) = str2double(char(T.Label(i)));
    end
    seedRowIdx(i) = rowIdx;
end

fprintf('Seed label distribution:\n');
for c = 0:9
    fprintf('  Class %d: %d\n', c, sum(seedLabels==c));
end

% ---------------------------------------------------------
% STEP 2b: Augment seed images (pixel space)
% ---------------------------------------------------------
numAugPerImage = 6;
numAug         = numSeed * numAugPerImage;
shiftPx        = 2;
noiseVar       = 0.002;

augPix    = zeros(numAug, pixDim, 'single');
augLabels = zeros(numAug, 1);

idx = 1;
for i = 1:numSeed
    imgF = reshape(seedPix(i,:), 28, 28);   % single [0,1]
    lbl  = seedLabels(i);

    variants{1} = imrotate(imgF,  5, 'bilinear', 'crop');
    variants{2} = imrotate(imgF, -5, 'bilinear', 'crop');
    variants{3} = imnoise(imgF, 'gaussian', 0, noiseVar);
    variants{4} = imtranslate(imgF, [ shiftPx,  0]);
    variants{5} = imtranslate(imgF, [0,  shiftPx]);
    variants{6} = imtranslate(imgF, [-shiftPx,  0]);

    for v = 1:numAugPerImage
        augPix(idx,:) = reshape(single(variants{v}), 1, []);
        augLabels(idx) = lbl;
        idx = idx + 1;
    end
end
fprintf('Augmentation done: %d images.\n', numAug);

% ---------------------------------------------------------
% STEP 2c: Assemble initial training set with sample weights
%
% As required by the assignment:
%   seed images (human-labelled) → weight = 100
%   augmented copies             → weight = 1
%   boundary images (added later)→ weight = 100
%   pseudo-labelled images       → weight = 1
% ---------------------------------------------------------
trainFeatures = [seedPix;    augPix    ];
trainLabels   = [seedLabels; augLabels ];
trainWeights  = [100*ones(numSeed,1); ones(numAug,1)];

fprintf('Training set: %d samples x %d features.\n', ...
        size(trainFeatures,1), size(trainFeatures,2));

% ---------------------------------------------------------
% STEP 3: SVM-1
%
% KernelScale: compute median pairwise distance on a
% subsample of the STANDARDIZED seed features.
% Standardize=true inside templateSVM handles the scaling,
% but we need to pre-standardize here just to estimate
% a good KernelScale.
% ---------------------------------------------------------
% Estimate KernelScale on standardized seed pixels
mu    = mean(seedPix, 1);
sigma = std(seedPix,  0, 1);
sigma(sigma == 0) = 1;   % avoid div-by-zero for constant pixels

seedStd = (seedPix - mu) ./ sigma;
sub     = seedStd(randperm(numSeed, min(100, numSeed)), :);
dists   = pdist(double(sub));
kernelScale = median(dists);
fprintf('\nEstimated KernelScale: %.4f\n', kernelScale);

t = templateSVM('KernelFunction', 'rbf',         ...
                'KernelScale',     kernelScale,   ...
                'BoxConstraint',   10,            ...
                'Standardize',     true);

fprintf('Training SVM-1...\n');
mdl = fitcecoc(trainFeatures, trainLabels, 'Learners', t, ...
               'Weights', trainWeights);

[labels, scores] = predict(mdl, X_all);
sorted_scores    = sort(scores, 2, 'descend');
margin           = sorted_scores(:,1) - sorted_scores(:,2);

fprintf('Prediction distribution after SVM-1:\n');
for c = 0:9, fprintf('  Class %d: %d\n', c, sum(labels==c)); end

initial_acc = check_accuracy(double(labels));
fprintf('Initial accuracy: %.2f%%\n\n', initial_acc * 100);

% ---------------------------------------------------------
% STEPS 4-7: Iterative Active Refinement + Self-Training
% ---------------------------------------------------------
N_uncertain     = 20;
N_per_class     = 50;    % 50 x 10 = 500 pseudo per iteration
max_iterations  = 20;
accuracy_target = 0.99;
conv_delta      = 0.001;

% Track which rows in X_all are already in the training set
labelledSet = false(numImages, 1);
for i = 1:numSeed
    labelledSet(seedRowIdx(i)) = true;
end

% Logging
log_accuracy = zeros(max_iterations, 1);
log_boundary = zeros(max_iterations, 1);
log_pseudo   = zeros(max_iterations, 1);
log_rejected = zeros(max_iterations, 1);
prev_accuracy    = initial_acc;
no_improve_count = 0;

fprintf('========================================\n');
fprintf(' Iterative refinement loop\n');
fprintf('========================================\n\n');

for iter = 1:max_iterations

    fprintf('--- Iteration %d ---\n', iter);

    % --------------------------------------------------
    % STEP 4: Human labels 20 most uncertain images
    % --------------------------------------------------
    unlabelledIdx  = find(~labelledSet);
    [~, sortedPos] = sort(margin(unlabelledIdx), 'ascend');
    nBoundary      = min(N_uncertain, length(sortedPos));
    boundaryIdx    = unlabelledIdx(sortedPos(1:nBoundary));

    % Show all 20 as a grid
    figure('Name', sprintf('Iter %d — Boundary Images', iter), ...
           'NumberTitle', 'off');
    for k = 1:nBoundary
        subplot(4, 5, k);
        imshow(reshape(X_all(boundaryIdx(k),:), 28, 28), []);
        title(sprintf('#%d p:%d', k, labels(boundaryIdx(k))), 'FontSize', 7);
    end
    sgtitle(sprintf('Iter %d — label in command window', iter));

    % Collect labels one by one with zoomed view
    humanLabels = zeros(nBoundary, 1);
    for k = 1:nBoundary
        figure(99); clf;
        imshow(reshape(X_all(boundaryIdx(k),:), 28, 28), [], ...
               'InitialMagnification', 800);
        title(sprintf('Image %d/%d  |  SVM predicted: %d', ...
              k, nBoundary, labels(boundaryIdx(k))));
        humanLabels(k) = input('  True label (0-9): ');
    end
    close(99);

    % Add boundary images with weight=100 (assignment requirement)
    trainFeatures = [trainFeatures; X_all(boundaryIdx,:)          ];
    trainLabels   = [trainLabels;   humanLabels                    ];
    trainWeights  = [trainWeights;  100*ones(nBoundary,1)          ];
    labelledSet(boundaryIdx) = true;
    log_boundary(iter) = nBoundary;
    fprintf('  Added %d boundary images (weight=100).\n', nBoundary);

    % --------------------------------------------------
    % STEP 5: Self-training — top 50 per class
    %
    % FIX: per-class 75th percentile threshold instead of
    % global — guarantees each class can contribute up to 50
    % images regardless of the global margin distribution.
    %
    % Threshold adapts to current accuracy:
    %   < 50%  → no threshold (rank-only), too early to filter
    %   >= 50% → per-class p75 threshold
    % --------------------------------------------------
    if prev_accuracy < 0.50
        use_threshold = false;
        thresh_label  = 'rank-only';
    else
        use_threshold = true;
        thresh_label  = 'per-class-p75';
    end

    newPseudoFeatures = [];
    newPseudoLabels   = [];
    numAdded    = 0;
    numRejected = 0;

    for c = 0:9
        % All unlabelled images predicted as class c
        classPool = find(~labelledSet & labels == c);
        if isempty(classPool), continue; end

        if use_threshold
            % Threshold computed per-class so each class gets a fair chance
            thresh_c    = prctile(margin(classPool), 75);
            above       = classPool(margin(classPool) >  thresh_c);
            below       = classPool(margin(classPool) <= thresh_c);
            numRejected = numRejected + length(below);
        else
            above = classPool;
        end

        if isempty(above), continue; end

        [~, s]   = sort(margin(above), 'descend');
        selected = above(s(1:min(N_per_class, length(s))));

        newPseudoFeatures = [newPseudoFeatures; X_all(selected,:)];
        newPseudoLabels   = [newPseudoLabels;   labels(selected) ];
        labelledSet(selected) = true;
        numAdded = numAdded + length(selected);
    end

    if ~isempty(newPseudoFeatures)
        trainFeatures = [trainFeatures; newPseudoFeatures                   ];
        trainLabels   = [trainLabels;   newPseudoLabels                     ];
        trainWeights  = [trainWeights;  ones(size(newPseudoFeatures,1), 1)  ];
    end

    log_pseudo(iter)   = numAdded;
    log_rejected(iter) = numRejected;
    fprintf('  [%s] Pseudo: %d added | %d rejected | Train size: %d\n', ...
            thresh_label, numAdded, numRejected, size(trainFeatures,1));

    % --------------------------------------------------
    % STEP 6: Retrain and evaluate
    %
    % FIX: recompute KernelScale from current labelled data
    % each iteration so the kernel adapts as the training
    % set grows. Sample 200 points max for speed.
    % --------------------------------------------------
    labelledRows  = find(labelledSet);
    subIdx        = labelledRows(randperm(length(labelledRows), min(200,length(labelledRows))));
    subData       = double(X_all(subIdx,:));
    mu_iter       = mean(subData, 1);
    sig_iter      = std(subData,  0, 1);
    sig_iter(sig_iter==0) = 1;
    subStd        = (subData - mu_iter) ./ sig_iter;
    dists_iter    = pdist(subStd);
    kernelScale   = median(dists_iter);

    t = templateSVM('KernelFunction', 'rbf',        ...
                    'KernelScale',     kernelScale,  ...
                    'BoxConstraint',   10,           ...
                    'Standardize',     true);

    fprintf('  KernelScale this iter: %.4f\n', kernelScale);
    fprintf('  Retraining SVM...\n');
    mdl = fitcecoc(trainFeatures, trainLabels, 'Learners', t, ...
                   'Weights', trainWeights);

    [labels, scores] = predict(mdl, X_all);
    sorted_scores    = sort(scores, 2, 'descend');
    margin           = sorted_scores(:,1) - sorted_scores(:,2);

    current_accuracy   = check_accuracy(double(labels));
    log_accuracy(iter) = current_accuracy;
    fprintf('  Accuracy: %.2f%%\n', current_accuracy * 100);

    dist_str = '';
    for c = 0:9
        dist_str = [dist_str sprintf('%d:%d ', c, sum(labels==c))];
    end
    fprintf('  Dist: %s\n\n', dist_str);

    % --------------------------------------------------
    % STEP 7: Stopping conditions
    %
    % FIX: don't stop on a single accuracy dip — require
    % 2 consecutive iterations without meaningful improvement
    % before declaring convergence.
    % --------------------------------------------------
    if current_accuracy >= accuracy_target
        fprintf('  >> Target %.0f%% reached!\n', accuracy_target * 100);
        log_accuracy = log_accuracy(1:iter);
        log_boundary = log_boundary(1:iter);
        log_pseudo   = log_pseudo(1:iter);
        log_rejected = log_rejected(1:iter);
        break;
    end

    gain = current_accuracy - prev_accuracy;
    if iter >= 2 && gain < conv_delta
        no_improve_count = no_improve_count + 1;
        fprintf('  >> No improvement (%d/2).\n', no_improve_count);
        if no_improve_count >= 2
            fprintf('  >> Converged after 2 consecutive non-improving iterations.\n');
            log_accuracy = log_accuracy(1:iter);
            log_boundary = log_boundary(1:iter);
            log_pseudo   = log_pseudo(1:iter);
            log_rejected = log_rejected(1:iter);
            break;
        end
    else
        no_improve_count = 0;
    end

    prev_accuracy = current_accuracy;
end

% ---------------------------------------------------------
% FINAL REPORT
% ---------------------------------------------------------
num_iters     = length(log_accuracy);
total_boundary = sum(log_boundary);
total_manual   = 300 + total_boundary;
manual_time_s  = total_manual * 10;

fprintf('\n========================================\n');
fprintf(' PIPELINE 2 — FINAL REPORT\n');
fprintf('========================================\n');
fprintf('Feature        : Raw pixels (784-dim)\n');
fprintf('KernelScale    : %.4f\n', kernelScale);
fprintf('BoxConstraint  : 10\n');
fprintf('Iterations     : %d\n',     num_iters);
fprintf('Final accuracy : %.2f%%\n', log_accuracy(end) * 100);
fprintf('Seed (manual)  : 300 images\n');
fprintf('Boundary images: %d\n',     total_boundary);
fprintf('Total manual   : %d images = %d s = %.1f min\n', ...
        total_manual, manual_time_s, manual_time_s / 60);

fprintf('\n%-4s %-10s %-10s %-13s %-10s\n', ...
        'Iter','Acc(%)', 'Boundary','Pseudo added','Rejected');
for i = 1:num_iters
    fprintf('%-4d %-10.2f %-10d %-13d %-10d\n', ...
            i, log_accuracy(i)*100, log_boundary(i), ...
            log_pseudo(i), log_rejected(i));
end

save('pipeline2_final.mat', 'mdl', 'labels', 'log_accuracy', ...
     'log_boundary', 'log_pseudo', 'log_rejected', 'kernelScale');
disp('Saved pipeline2_final.mat');