clear; close all; clc;

% Base directories
baseDir    = 'AMOE-master/AMOE-master/AMOE/imagesgrabcut/';
imgDir     = fullfile(baseDir, 'images');
slicDir    = fullfile(baseDir, 'slic');
g2bDir     = fullfile(baseDir, 'g2b');
markerDir  = fullfile(baseDir, 'markers');
outDir     = fullfile(baseDir, 'SSNCut_output');

% Get image file list, skipping directories
imgFiles = dir(fullfile(imgDir, '*.*'));
imgFiles = imgFiles(~[imgFiles.isdir]);

% Set starting index
startIdx = 1;
numImgs = length(imgFiles);

fprintf('Processing %d images (starting from index %d)...\n', numImgs - startIdx + 1, startIdx);

% Create output directory if it doesn't exist
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

    % Extract RGB marker masks
    R = markerImg(:,:,1);
    G = markerImg(:,:,2);
    B = markerImg(:,:,3);

    class1Mask = (G == 255 & R == 0 & B == 0);  % Green
    class2Mask = (B == 255 & R == 0 & G == 0);  % Blue
    C = cat(3, class1Mask, class2Mask);

    % Load gPb and superpixels
    g2bPath = fullfile(g2bDir, [imgName, '.mat']);
    slicPath = fullfile(slicDir, [imgName, '.mat']);

    if ~exist(g2bPath, 'file') || ~exist(slicPath, 'file')
        warning('Missing .mat files for %s. Skipping...', imgName);
        continue;
    end

    load(g2bPath);  % Loads 'gPb'
    load(slicPath); % Loads 'spSegs'

    % Compute centroids
    numSuperpixels = max(spSegs(:)) + 1;
    [r, c] = ndgrid(1:size(img,1), 1:size(img,2));
    meanPosition = zeros(numSuperpixels, 2);

    for ii = 1:numSuperpixels
        mask = (spSegs == (ii-1));
        meanPosition(ii, :) = [mean(r(mask)), mean(c(mask))];
    end

    % Constrained superpixels
    CSupPix = false(numSuperpixels, 2);
    for ii = 1:numSuperpixels
        mask = (spSegs == (ii-1));
        for jj = 1:2
            if any(C(:,:,jj) & mask, 'all')
                CSupPix(ii,jj) = true;
            end
        end
    end

    % Build adjacency graph
    sigma = 0.1;
    rad = max(size(img,1), size(img,2)) / 10;
    A = adjacencyMatrixGPbSuperpixels(meanPosition, gPb, rad, sigma);

    % ----------- Safety Checks ----------- %
    if any(isnan(A(:))) || any(isinf(A(:)))
        warning('Adjacency matrix A contains NaN or Inf. Skipping image: %s', imagename);
        continue;
    end

    if nnz(A) == 0
        warning('Adjacency matrix A is empty (all zeros). Skipping image: %s', imagename);
        continue;
    end

    % Check graph connectivity
    G = graph(A);
    bins = conncomp(G);
    largestComponentRatio = max(histcounts(bins)) / size(A,1);
    if largestComponentRatio < 0.5
        warning('Graph A is poorly connected. Skipping image: %s', imagename);
        continue;
    end

    % Create ML (must-link) and CL (cannot-link) constraints
    constrInd = zeros(numSuperpixels,1);
    MLcell = cell(2,1);
    for ii = 1:2
        ind = find(CSupPix(:,ii));
        constrInd(ind) = ii;
        MLcell{ii} = completeGraph(ind);
    end
    ML = cat(1, MLcell{:});

    ind1 = find(CSupPix(:,1));
    ind2 = find(CSupPix(:,2));
    CL = [repmat(ind1(:), [1 numel(ind2)]), repmat(ind2(:)', [numel(ind1), 1])];
    CL = reshape(CL, [], 2);

    if isempty(ML) || isempty(CL)
        warning('Not enough ML or CL constraints. Skipping image: %s', imagename);
        continue;
    end

    % Compute gamma safely
    numML = nchoosek(sum(C(:,:,1), 'all'), 2) + nchoosek(sum(C(:,:,2), 'all'), 2);
    numCL = sum(C(:,:,1), 'all') * sum(C(:,:,2), 'all');
    gamma = 100 * (4 * sum(A(:))) ./ [max(numML,1), max(numCL,1)];
    gamma = min(gamma, 1e6);  % Upper bound
    gamma = max(gamma, 1e-3); % Lower bound

    % Semi-supervised NCut
    try
        YSS = semiSupervisedNormCut(A, 2, ML, gamma(1), CL, gamma(2), [], constrInd);
        YSS = YSS .* sign(YSS(ind1(1)));  % Fix sign ambiguity
    catch ME
        warning('semiSupervisedNormCut failed for %s: %s', imagename, ME.message);
        continue;
    end
    
    try
        % Interpolate result to image grid
        segmentation = interpSpectraSupPix(img, spSegs, meanPosition, 1, YSS);
    catch ME
        warning('semiSupervisedNormCut failed for %s: %s', imagename, ME.message);
        continue;
    end
    binaryMask = segmentation > 0;

    % Save binary mask
    outPath = fullfile(outDir, [imgName '.bmp']);
    imwrite(binaryMask, outPath);
    fprintf('Saved: %s\n', outPath);
end
