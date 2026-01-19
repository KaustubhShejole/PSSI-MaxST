%% Minimal SSNCut Binary Segmentation Output

% Set up paths
exampleDataDir = 'AMOE-master/AMOE-master/AMOE/imagesgrabcut/';
imgName = 'banana1';

% Load the main image
imgFile = fullfile(exampleDataDir, 'images', [imgName, '.bmp']);
img = imread(imgFile);

% Load the marker image
markerImg = imread(fullfile(exampleDataDir, 'markers', [imgName, '_marker.bmp']));

% Extract RGB channels
R = markerImg(:,:,1);
G = markerImg(:,:,2);
B = markerImg(:,:,3);

% Class 1: Green (0, 255, 0)
class1Mask = G == 255 & R == 0 & B == 0;

% Class 2: Blue (0, 0, 255)
class2Mask = B == 255 & R == 0 & G == 0;

% Combine into 3D logical array
C = cat(3, class1Mask, class2Mask);


% Load precomputed gPb and superpixels
load(fullfile(exampleDataDir, 'g2b', [imgName, '.mat']));
% Load precomputed superpixels
load(fullfile(exampleDataDir, 'slic', [imgName, '.mat']));



% Compute superpixel centroids
numSuperpixels = max(spSegs(:)) + 1;
[r, c] = ndgrid(1:size(img,1), 1:size(img,2));
meanPosition = zeros(numSuperpixels, 2);
for ii = 1:numSuperpixels
    mask = (spSegs == (ii-1));
    meanPosition(ii, :) = [mean(r(mask)), mean(c(mask))];
end

% Identify constrained superpixels
CSupPix = false(numSuperpixels, 2);
for ii = 1:numSuperpixels
    mask = (spSegs == (ii-1));
    for jj = 1:2
        if any(C(:,:,jj) & mask, 'all')
            CSupPix(ii,jj) = true;
        end
    end
end

% Build graph
sigma = 0.1;
rad = max(size(img,1), size(img,2));
A = adjacencyMatrixGPbSuperpixels(meanPosition, gPb, rad, sigma);

% Create ML and CL constraints
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

% Compute gamma
numML = nchoosek(sum(C(:,:,1), 'all'), 2) + nchoosek(sum(C(:,:,2), 'all'), 2);
numCL = sum(C(:,:,1), 'all') * sum(C(:,:,2), 'all');
gamma = 100 * (4 * sum(A(:))) ./ [numML, numCL];

% Run Semi-Supervised NCut
YSS = semiSupervisedNormCut(A, 2, ML, gamma(1), CL, gamma(2), [], constrInd);
YSS = YSS .* sign(YSS(ind1(1)));  % Fix sign ambiguity

% Interpolate result to image grid
segmentation = interpSpectraSupPix(img, spSegs, meanPosition, 1, YSS);

% Threshold to binary mask (foreground vs background)
binaryMask = segmentation > 0;

% Display binary segmentation mask
imshow(binaryMask); title('Final Binary Segmentation Mask');

% Optional: Save mask
% imwrite(binaryMask, 'SSNCut_binary_mask.png');

outDir = fullfile(exampleDataDir, 'output_SSNCut');
if ~exist(outDir, 'dir'), mkdir(outDir); end
imwrite(binaryMask, fullfile(outDir, [imgName(1:end-4) '.bmp']));
disp(['Saved: ' imgName]);