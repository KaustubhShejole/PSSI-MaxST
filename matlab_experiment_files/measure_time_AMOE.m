clear all; close all;
addpath('.');
addpath('Ncut_9');
addpath('MatlabHPF');
addpath('imagesgrabcut');
addpath('algorithms');
addpath('onecut');

exampleDataDir = 'images250';

ims = dir([exampleDataDir '/images/*.*']);

outDir = fullfile(exampleDataDir, 'amoe_amoe_scribbles');
markerDir = fullfile(exampleDataDir, 'amoe_markers');

% Create output directory
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Initialize timing and logging
time_list = [];
image_names = {};
total_tic = tic;

for imgindex = 3:53
    

    imagename = ims(imgindex).name;
    imgFile = fullfile(exampleDataDir, ['/images/' imagename]);
    img = imread(imgFile);
    img1 = img;

    % Display original image
    figure; imshow(img); title('Original image');

    nei = 1;           % 0: 4-neighbors, 1: 8-neighbors
    scale = 1.0;       % image resize
    lambda = 2e-10;    % parameter for unitary
    imgName = 0;

    ref_name = fullfile(markerDir, [ims(imgindex).name(1:end-4) '_marker.bmp']);
    [K, labels, idx] = seed_generation(ref_name, scale);

    [X, Y, Z] = size(img);
    N = X * Y;

    boximage = fullfile(exampleDataDir, ['/box/' ims(imgindex).name(1:end-4) '_box.bmp']);
    box = imread(boximage);
    boxone = box(:, :, 1);
    boxsize = sum(boxone(:) == 0);

    image_tic = tic;  % Start timer
    % Set energy parameters
    lambda = 9.0;
    beta_prime = 0.9;
    nei = 1;
    channelstep = 4;
    INFTY = 100000000;

    [points, edges] = lattice(X, Y, nei);
    imgVals = reshape(img, N, Z);
    imgVals = double(imgVals);
    di = (imgVals(edges(:,1),1) - imgVals(edges(:,2),1)).^2 + ...
         (imgVals(edges(:,1),2) - imgVals(edges(:,2),2)).^2 + ...
         (imgVals(edges(:,1),3) - imgVals(edges(:,2),3)).^2;

    sigma_square = sum(di(:,1)) / length(edges);
    colorlabel = rgb2indeximg(img, channelstep);
    [colorbinnum, compacthist, colorlabelchange] = getcompactlabel(colorlabel);
    l1penalty = getl1penalty(colorlabelchange, box);

    beta = boxsize / l1penalty * beta_prime;

    ROI = true(X, Y);
    addsmoothnessterm_weight = addsmoothnessterm(edges, points, img, imgVals, lambda, ROI, sigma_square);
    [edges_addauxnode, weight_addauxnode] = addl1separationterm(edges, addsmoothnessterm_weight, colorlabelchange, beta, ROI);
    W = adjacency(edges_addauxnode, weight_addauxnode, N + colorbinnum);

    if Z > 1
        I = double(rgb2gray(img));
    else
        I = double(img);
    end
    [w, imageEdges] = ICgraph(I);
    w(N + colorbinnum, N + colorbinnum) = 0;
    W = W + w;

    if imgName == 0
        for k = 1:K
            seedsInd{k} = idx(labels == k);
        end
    end

    source = seedsInd{1};
    sink = seedsInd{2};
    sink2 = find(boxone == 255);
    unkown = find(boxone == 0);

    Nsum = N + colorbinnum;
    W(Nsum + 1, source) = INFTY;
    W(sink, Nsum + 2) = INFTY;
    W(sink2, Nsum + 2) = INFTY;
    W(Nsum + 2, Nsum + 2) = 0;
    W = W - diag(diag(W));

    [value, cut] = hpf(W, Nsum + 1, Nsum + 2);
    cut1 = cut(1:N);
    seg = reshape(cut1, X, Y);
    figure; imagesc(seg);

    % Overlay segmentation on original image
    bw = edge(seg, 0.01);
    [i, j] = find(bw);
    edgeindex = [i, j];
    fgColor = [255, 0, 0];
    imgseg = img1;

    [nr, nc] = size(edgeindex(:, 1));
    for i = 1:3
        for j = 1:nr
            imgseg(edgeindex(j, 1), edgeindex(j, 2), i) = fgColor(i);
        end
    end
    figure; imshow(imgseg);

    % Save results
    % outputfile = fullfile(outDir, [ims(imgindex).name(1:end-4) '_hpfseg.bmp']);
    % imwrite(imgseg, outputfile);
    % outputfile = fullfile(outDir, [ims(imgindex).name(1:end-4) '.bmp']);
    % imwrite(seg, outputfile);

    % Store timing info
    elapsed_time = toc(image_tic);
    time_list = [time_list; elapsed_time];
    image_names{end + 1, 1} = imagename;
    fprintf('Processed %s in %.2f seconds\n', imagename, elapsed_time);

    close all;
end

% Final timing summary
total_time = toc(total_tic);
mean_time = mean(time_list);
std_time = std(time_list);
min_time = min(time_list);
max_time = max(time_list);

fprintf('\n=== Segmentation Timing Summary ===\n');
fprintf('Total time: %.2f seconds\n', total_time);
fprintf('Number of images: %d\n', length(time_list));
fprintf('Mean time: %.2f seconds\n', mean_time);
fprintf('Standard deviation: %.2f seconds\n', std_time);
fprintf('Minimum time: %.2f seconds\n', min_time);
fprintf('Maximum time: %.2f seconds\n', max_time);

% Write CSV
T = table(image_names, time_list, 'VariableNames', {'ImageName', 'TimeSeconds'});
writetable(T, 'amoe_time_images250.csv');