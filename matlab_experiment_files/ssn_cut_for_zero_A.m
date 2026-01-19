clear; close all; clc;

% Base directories
baseDir    = 'AMOE-master/AMOE-master/AMOE/images250/';
imgDir     = fullfile(baseDir, 'images');
g2bDir     = fullfile(baseDir, 'g2b');
markerDir  = fullfile(baseDir, 'ssncutmarker');
outDir     = fullfile(baseDir, 'output_SSNCut');

% Get image file list
imgFiles = dir(fullfile(imgDir, '*.*'));
imgFiles = imgFiles(~[imgFiles.isdir]);

% Settings
startIdx = 24;
numImgs = length(imgFiles);
fprintf('Processing %d images...\n', numImgs - startIdx + 1);

% Create output directory
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Loop over images
for imgindex = startIdx:numImgs
    fprintf('[%d/%d] ', imgindex, numImgs);
    imagename = imgFiles(imgindex).name;
    [~, imgName, ~] = fileparts(imagename);
    
    % Read image and marker
    imgPath = fullfile(imgDir, imagename);
    markerPath = fullfile(markerDir, [imgName '_marker.bmp']);
    
    if ~exist(markerPath, 'file')
        warning('Marker file not found for %s. Skipping...', imagename);
        continue;
    end
    
    img = imread(imgPath);
    markerImg = imread(markerPath);
    
    % Extract marker masks
    class1Mask = (markerImg(:,:,2) == 255 & markerImg(:,:,1) == 0 & markerImg(:,:,3) == 0); % Green
    class2Mask = (markerImg(:,:,3) == 255 & markerImg(:,:,1) == 0 & markerImg(:,:,2) == 0); % Blue
    C = cat(3, class1Mask, class2Mask);
    
    % Load gPb boundary map
    g2bPath = fullfile(g2bDir, [imgName, '.mat']);
    if ~exist(g2bPath, 'file')
        warning('gPb file not found for %s. Skipping...', imgName);
        continue;
    end
    load(g2bPath); % Loads 'gPb'
    
    % Compute superpixels using MATLAB (0-indexed labels)
    numDesired = 250;
    [spLabels, numActual] = superpixels(img, numDesired);
    spSegs = double(spLabels) - 1; % Convert to 0-indexed
    numSuperpixels = max(spSegs(:)) + 1; % Includes all labels (0 to numActual-1)
    
    % Compute centroids
    [rows, cols] = size(spSegs);
    [r, c] = ndgrid(1:rows, 1:cols);
    meanPosition = zeros(numSuperpixels, 2);
    for ii = 1:numSuperpixels
        mask = (spSegs == (ii-1));
        meanPosition(ii, :) = [mean(r(mask)), mean(c(mask))];
    end
    
    % Identify constrained superpixels (overlap with markers)
    CSupPix = false(numSuperpixels, 2);
    for ii = 1:numSuperpixels
        mask = (spSegs == (ii-1));
        for jj = 1:2
            if any(C(:,:,jj) & mask, 'all')
                CSupPix(ii,jj) = true;
            end
        end
    end
    
    % Build adjacency matrix with fallback for connectivity
    sigma = 0.1;
    rad_initial = max(rows, cols) / 10;
    A = adjacencyMatrixGPbSuperpixels(meanPosition, gPb, rad_initial, sigma);
    
    % Fallback if adjacency matrix is empty
    if nnz(A) == 0
        rad_fallback = max(rows, cols) / 5; % Increase radius
        A = adjacencyMatrixGPbSuperpixels(meanPosition, gPb, rad_fallback, sigma);
        if nnz(A) == 0
            warning('Adjacency matrix still empty after fallback. Skipping: %s', imagename);
            continue;
        end
    end
    
    % Graph connectivity check
    G = graph(A);
    bins = conncomp(G);
    if max(bins) < 2 || max(histcounts(bins)) / numSuperpixels < 0.5
        warning('Graph connectivity issue. Skipping: %s', imagename);
        continue;
    end
    
    % Create constraints (ML: Must-Link, CL: Cannot-Link)
    constrInd = zeros(numSuperpixels, 1);
    ML = []; CL = [];
    for class = 1:2
        idx = find(CSupPix(:, class));
        constrInd(idx) = class;
        if numel(idx) > 1
            ML = [ML; nchoosek(idx, 2)]; % Complete graph for class
        end
    end
    class1Idx = find(CSupPix(:,1));
    class2Idx = find(CSupPix(:,2));
    if ~isempty(class1Idx) && ~isempty(class2Idx)
        [idx1, idx2] = ndgrid(class1Idx, class2Idx);
        CL = [idx1(:), idx2(:)];
    end
    
    if isempty(ML) || isempty(CL)
        warning('Insufficient constraints. Skipping: %s', imagename);
        continue;
    end
    
    % Compute gamma for constraints
    numML = size(ML, 1);
    numCL = size(CL, 1);
    gamma = 100 * (4 * sum(A(:))) ./ [max(numML,1), max(numCL,1)];
    gamma = min(gamma, 1e6); % Cap gamma to avoid numerical issues
    
    % Semi-supervised Ncut
    try
        YSS = semiSupervisedNormCut(A, 2, ML, gamma(1), CL, gamma(2), [], constrInd);
        if ~isempty(class1Idx)
            YSS = YSS .* sign(YSS(class1Idx(1))); % Resolve sign ambiguity
        end
    catch ME
        warning('semiSupervisedNormCut failed: %s. Skipping...', ME.message);
        continue;
    end
    
    % Generate segmentation mask
    segmentation = interpSpectraSupPix(img, spSegs, meanPosition, 1, YSS);
    binaryMask = segmentation > 0;
    
    % Save result
    outPath = fullfile(outDir, [imgName '.bmp']);
    imwrite(binaryMask, outPath);
    fprintf('Saved: %s\n', outPath);
end