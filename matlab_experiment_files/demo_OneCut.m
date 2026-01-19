% OneCut With Scribbles
clear; close all; clc;

% Add paths to helper functions
addpath('.');
addpath('Ncut_9');
addpath('MatlabHPF');
addpath('imagesgrabcut');
addpath('algorithms');
addpath('onecut');

% Configuration
exampleDataDir = 'imagesgrabcut';
ims = dir(fullfile(exampleDataDir, 'images', '*.*'));

% Parameters
lambda = 9.0;
beta_prime = 0.9;
nei = 1;  % 8-connectivity
channelstep = 1; % 256^3 bins
INFTY = 1e8;

for imgindex = 3:length(ims)
    imagename = ims(imgindex).name;
    fprintf('Processing: %s\n', imagename);

    img = imread(fullfile(exampleDataDir, 'images', imagename));
    img1 = img;

    box = imread(fullfile(exampleDataDir, 'box', [imagename(1:end-4) '_box.bmp']));
    boxone = box(:,:,1);
    boxsize = sum(boxone(:) == 0);

    marker = imread(fullfile(exampleDataDir, 'markers', [imagename(1:end-4) '_marker.bmp']));

    [X, Y, Z] = size(img);
    N = X * Y;
    img = double(img);

    %% Display
    figure; imshow(uint8(img)); title('Original image');

    %% Seed Generation (from scribbles)
    scale = 1.0;
    [K, labels, idx] = seed_generation(fullfile(exampleDataDir, 'markers', [imagename(1:end-4) '_marker.bmp']), scale);

    %% Graph Construction
    [points, edges] = lattice(X, Y, nei);
    imgVals = reshape(img, N, Z);
    di = sum((imgVals(edges(:,1),:) - imgVals(edges(:,2),:)).^2, 2);
    sigma_square = mean(di);

    %% Smoothness Term
    ROI = true(X, Y);
    smooth_weights = addsmoothnessterm(edges, points, uint8(img), imgVals, lambda, ROI, sigma_square);

    %% L1 Color Separation Term
    colorlabel = rgb2indeximg(uint8(img), channelstep);
    [colorbinnum, ~, colorlabelchange] = getcompactlabel(colorlabel);
    l1penalty = getl1penalty(colorlabelchange, box);
    beta = (boxsize / l1penalty) * beta_prime;

    [edges2, weights2] = addl1separationterm(edges, smooth_weights, colorlabelchange, beta, ROI);

    %% Adjacency Matrix
    W = adjacency(edges2, weights2, N + colorbinnum);
    W(N + colorbinnum, N + colorbinnum) = 0;

    %% IC Edge Graph Term
    if Z > 1
        I = double(rgb2gray(uint8(img)));
    else
        I = double(img);
    end
    [w, ~] = ICgraph(I);
    w(N + colorbinnum, N + colorbinnum) = 0;
    W = W + w;

    %% Add Source/Sink from Scribbles and Box
    for k = 1:K
        seeds{k} = idx(labels == k);
    end
    source = seeds{1};
    sink = seeds{2};
    sink2 = find(box(:,:,1) == 255);

    Nsum = N + colorbinnum;
    W(Nsum + 1, source) = INFTY;
    W(sink, Nsum + 2) = INFTY;
    W(sink2, Nsum + 2) = INFTY;
    W(Nsum + 2, Nsum + 2) = 0;
    W = W - diag(diag(W));

    %% Min-Cut using HPF
    [~, cut] = hpf(W, Nsum + 1, Nsum + 2);
    seg = reshape(cut(1:N), X, Y);

    %% Overlay on original
    bw = edge(seg, 0.01);
    [i, j] = find(bw);
    fgColor = [255, 0, 0];
    imgseg = img1;
    for c = 1:3
        for p = 1:length(i)
            imgseg(i(p), j(p), c) = fgColor(c);
        end
    end

    %% Save Outputs
    outDir = fullfile(exampleDataDir, 'output_OneCut_Adapted');
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    imwrite(seg, fullfile(outDir, [imagename(1:end-4) '_seg.bmp']));
    imwrite(imgseg, fullfile(outDir, [imagename(1:end-4) '_overlay.bmp']));

    close all;
end