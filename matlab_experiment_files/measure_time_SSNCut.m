clear; close all; clc;

% Base directories
baseDir    = 'AMOE-master/AMOE-master/AMOE/images250/';
imgDir     = fullfile(baseDir, 'images');
slicDir    = fullfile(baseDir, 'slic');
g2bDir     = fullfile(baseDir, 'g2b');
markerDir  = fullfile(baseDir, 'ssncutmarker');
outDir     = fullfile(baseDir, 'SSNCut_output');

% Get image file list, skipping directories
imgFiles = dir(fullfile(imgDir, '*.*'));
imgFiles = imgFiles(~[imgFiles.isdir]);

% Index settings
startIdx = 1;
numImgs = 10;

fprintf('Processing %d images (starting from index %d)...\n', numImgs - startIdx + 1, startIdx);

% Ensure output directory exists
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Initialize timers
time_list = [];
image_names = {};
total_tic = tic;

% Loop over images
for imgindex = startIdx:numImgs
    image_tic = tic;  % Start timer for this image

    imagename = imgFiles(imgindex).name;
    [~, imgName, ~] = fileparts(imagename);
    fprintf('[%d/%d] Processing %s\n', imgindex, numImgs, imagename);

    try
        %% Load original image
        imgFile = fullfile(imgDir, imagename);
        img = imread(imgFile);
        originalSize = size(img);

        %% Downsample
        scale = 1.0;
        imgSmall = imresize(img, scale);

        %% Save temp image
        tmpImgPath = fullfile(tempdir, [imgName, '_tmp.jpg']);
        imwrite(imgSmall, tmpImgPath);

        %% Run globalPb
        fprintf('    ⏳ Running globalPb...\n');
        gPb_orient = globalPb(tmpImgPath);
        gPbSmall = max(gPb_orient, [], 3);
        gPb = imresize(gPbSmall, [originalSize(1), originalSize(2)], 'bilinear');
        gPbPath = fullfile(g2bDir, [imgName '.mat']);
        save(gPbPath, 'gPb');
        fprintf('    💾 Saved gPb\n');

        %% Compute SLIC superpixels
        cform = makecform('srgb2lab');
        imlab = applycform(img, cform);
        regionSize = 10;
        regularizer = 10000;
        spSegs = vl_slic(single(imlab), regionSize, regularizer);
        save(fullfile(slicDir, [imgName, '.mat']), 'spSegs');

    catch ME
        fprintf('⚠️  Error in preprocessing %s: %s\n', imgName, ME.message);
        continue;
    end

    % Read image and marker
    imgPath = fullfile(imgDir, imagename);
    markerPath = fullfile(markerDir, [imgName '_marker.bmp']);
    if ~exist(markerPath, 'file')
        warning('Marker file not found for %s. Skipping...', imagename);
        continue;
    end

    img = imread(imgPath);
    markerImg = imread(markerPath);
    R = markerImg(:,:,1); G = markerImg(:,:,2); B = markerImg(:,:,3);
    class1Mask = (G == 255 & R == 0 & B == 0);
    class2Mask = (B == 255 & R == 0 & G == 0);
    C = cat(3, class1Mask, class2Mask);

    g2bPath = fullfile(g2bDir, [imgName, '.mat']);
    slicPath = fullfile(slicDir, [imgName, '.mat']);
    if ~exist(g2bPath, 'file') || ~exist(slicPath, 'file')
        warning('Missing .mat files for %s. Skipping...', imgName);
        continue;
    end
    load(g2bPath); load(slicPath);

    numSuperpixels = max(spSegs(:)) + 1;
    [r, c] = ndgrid(1:size(img,1), 1:size(img,2));
    meanPosition = zeros(numSuperpixels, 2);
    for ii = 1:numSuperpixels
        mask = (spSegs == (ii-1));
        meanPosition(ii, :) = [mean(r(mask)), mean(c(mask))];
    end

    CSupPix = false(numSuperpixels, 2);
    for ii = 1:numSuperpixels
        mask = (spSegs == (ii-1));
        for jj = 1:2
            if any(C(:,:,jj) & mask, 'all')
                CSupPix(ii,jj) = true;
            end
        end
    end

    sigma = 0.1;
    rad = max(size(img,1), size(img,2)) / 10;
    try
        A = adjacencyMatrixGPbSuperpixels(meanPosition, gPb, rad, sigma);
    catch ME
        warning('SSNCut failed for %s: %s', imagename, ME.message);
        continue;
    end

    if any(isnan(A(:))) || any(isinf(A(:))) || nnz(A) == 0
        warning('Invalid adjacency matrix for %s. Skipping...', imagename);
        continue;
    end

    G = graph(A);
    bins = conncomp(G);
    if max(histcounts(bins)) / size(A,1) < 0.5
        warning('Graph poorly connected for %s. Skipping...', imagename);
        continue;
    end

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
        warning('Insufficient constraints for %s. Skipping...', imagename);
        continue;
    end

    numML = nchoosek(sum(C(:,:,1), 'all'), 2) + nchoosek(sum(C(:,:,2), 'all'), 2);
    numCL = sum(C(:,:,1), 'all') * sum(C(:,:,2), 'all');
    gamma = 100 * (4 * sum(A(:))) ./ [max(numML,1), max(numCL,1)];
    gamma = min(gamma, 1e6);
    gamma = max(gamma, 1e-3);

    try
        YSS = semiSupervisedNormCut(A, 2, ML, gamma(1), CL, gamma(2), [], constrInd);
        YSS = YSS .* sign(YSS(ind1(1)));
    catch ME
        warning('SSNCut failed for %s: %s', imagename, ME.message);
        continue;
    end

    try
        segmentation = interpSpectraSupPix(img, spSegs, meanPosition, 1, YSS);
    catch ME
        warning('Interpolation failed for %s: %s', imagename, ME.message);
        continue;
    end

    binaryMask = segmentation > 0;
    outPath = fullfile(outDir, [imgName '.bmp']);
    imwrite(binaryMask, outPath);
    fprintf('✅ Saved: %s\n', outPath);

    % Record timing
    elapsed_time = toc(image_tic);
    time_list = [time_list; elapsed_time];
    image_names{end+1,1} = imagename;
    fprintf('⏱️  Time: %.2f seconds\n', elapsed_time);
end

% Final summary
total_time = toc(total_tic);
mean_time = mean(time_list);
std_time = std(time_list);
min_time = min(time_list);
max_time = max(time_list);

fprintf('\n=== Segmentation Timing Summary ===\n');
fprintf('Total time: %.2f seconds\n', total_time);
fprintf('Images processed: %d\n', length(time_list));
fprintf('Mean time: %.2f s, Std: %.2f s, Min: %.2f s, Max: %.2f s\n', mean_time, std_time, min_time, max_time);

% Save timing data to CSV
T = table(image_names, time_list, 'VariableNames', {'ImageName', 'TimeSeconds'});
writetable(T, 'ssncut_time_images250.csv');
fprintf('📄 Timing data saved to amoe_time_images250.csv\n');