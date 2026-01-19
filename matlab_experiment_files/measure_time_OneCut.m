clear all; close all; clc;

% Add required paths
addpath('.');
addpath('Ncut_9');
addpath('MatlabHPF');
addpath('imagesgrabcut');
addpath('algorithms');
addpath('onecut');

% Configuration
exampleDataDir = 'images250';

% Parameters
lambda = 9.0;
beta_prime = 0.9;
nei = 1;                  % 8-neighbors
channelstep = 1;          % Color quantization (e.g. 256^3 bins)
INFTY = 1e8;

% Output directory
outDir = fullfile(exampleDataDir, 'onecut_onecut_scribbles');
markerDir = fullfile(exampleDataDir, 'onecutmarker_gb');
if ~exist(outDir, 'dir'), mkdir(outDir); end

ims = dir([exampleDataDir '/images/*.*']);
startidx = 3;
endidx = 200;

% Timing setup
time_list = [];
image_names = {};
total_tic = tic;

for imgindex = startidx:endidx

    imagename = ims(imgindex).name;
    fprintf('\nProcessing %s...\n', imagename);

    % === Load image and associated files ===
    imgFile = fullfile(exampleDataDir,['/images/' imagename]);
    img = imread(imgFile);
    box = imread(fullfile(exampleDataDir, 'box', [imagename(1:end-4) '_box.bmp']));
    marker = imread(fullfile(markerDir, [imagename(1:end-4) '_marker.bmp']));
    img_orig = img;

    [X, Y, Z] = size(img);
    N = X * Y;
    img = double(img);

    % === Get foreground/background seeds from marker ===
    [K, labels, idx] = seed_generation(fullfile(markerDir, [imagename(1:end-4) '_marker.bmp']), 1.0);
    seeds = cell(K, 1);
    for k = 1:K
        seeds{k} = idx(labels == k);
    end
    
    image_tic = tic;
    % === Construct pixel grid graph ===
    [points, edges] = lattice(X, Y, nei);
    imgVals = reshape(img, N, Z);
    diff = sum((imgVals(edges(:,1),:) - imgVals(edges(:,2),:)).^2, 2);
    sigma_square = mean(diff);
    smooth_weights = addsmoothnessterm(edges, points, uint8(img), imgVals, lambda, true(X,Y), sigma_square);

    % === Color binning for L1 separation ===
    colorlabel = rgb2indeximg(uint8(img), channelstep);
    [colorbinnum, ~, colorlabelchange] = getcompactlabel(colorlabel);

    % === Compute L1 penalty weight ===
    boxmask = box(:,:,1) == 0;
    boxsize = sum(boxmask(:));
    l1penalty = getl1penalty(colorlabelchange, box);
    beta = (boxsize / l1penalty) * beta_prime;

    % === Add L1 separation edges ===
    [edges2, weights2] = addl1separationterm(edges, smooth_weights, colorlabelchange, beta, true(X,Y));

    % === Adjacency Matrix ===
    W = adjacency(edges2, weights2, N + colorbinnum);
    W(N + colorbinnum, N + colorbinnum) = 0;

    % === Hard constraints from scribbles and box ===
    fg = seeds{1};                          % foreground scribbles
    bg = seeds{2};                          % background scribbles
    box_bg = find(box(:,:,1) == 255);       % background outside box

    Nsum = N + colorbinnum;
    S = Nsum + 1;  % source
    T = Nsum + 2;  % sink

    % Expand W to size (N + bins + 2) before adding edges
    W(S, 1) = 0;  % pre-allocate
    W(T, 1) = 0;

    % Add terminal edges
    W(S, fg) = INFTY;
    W(bg, T) = INFTY;
    W(box_bg, T) = INFTY;

    % Zero diagonal (safe operation)
    W = W - spdiags(diag(W), 0, size(W,1), size(W,2));

    % === Run HPF Min-Cut ===
    fprintf('Running min-cut...\n');
    [~, cut] = hpf(W, S, T);
    seg = reshape(cut(1:N), X, Y);

    % === Overlay segmentation edges on image ===
    bw = edge(seg, 0.01);
    overlay = uint8(img_orig);
    for c = 1:3
        channel = overlay(:,:,c);
        channel(bw) = 255 * (c == 1);  % red edge
        overlay(:,:,c) = channel;
    end

    % === Save segmentation and overlay ===
    %seg_path = fullfile(outDir, [imagename(1:end-4) '.bmp']);
    %overlay_path = fullfile(outDir, [imagename(1:end-4) '_overlay.bmp']);
    %imwrite(seg, seg_path);
    %imwrite(overlay, overlay_path);

    % === Time tracking ===
    elapsed_time = toc(image_tic);
    time_list = [time_list; elapsed_time];
    image_names{end+1, 1} = imagename;
    fprintf('Processed %s in %.2f seconds\n', imagename, elapsed_time);
end

% === Final Timing Summary ===
total_time = toc(total_tic);
mean_time = mean(time_list);
std_time = std(time_list);
min_time = min(time_list);
max_time = max(time_list);

fprintf('\n=== Pure OneCut Timing Summary ===\n');
fprintf('Total time: %.2f seconds\n', total_time);
fprintf('Images processed: %d\n', length(time_list));
fprintf('Mean time: %.2f seconds\n', mean_time);
fprintf('Standard deviation: %.2f seconds\n', std_time);
fprintf('Minimum time: %.2f seconds\n', min_time);
fprintf('Maximum time: %.2f seconds\n', max_time);

% === Write time log to CSV ===
T = table(image_names, time_list, 'VariableNames', {'ImageName', 'TimeSeconds'});
writetable(T, 'onecut_ssncut_scribbles_time.csv');

fprintf('\n✅ All images processed and timing saved to onecut_ssncut_scribbles_time.csv.\n');