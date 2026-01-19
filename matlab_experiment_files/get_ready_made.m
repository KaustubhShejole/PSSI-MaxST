%% Batch Processing for SSNCut Preprocessing (All Files, Skipping `.` and `..`)
% Uncomment below if VLFeat is needed
% run('vlfeat/vlfeat-0.9.21/toolbox/vl_setup.m');

% Set up paths
baseDir = 'AMOE-master/AMOE-master/AMOE/imagesgrabcut/';
fromDir = 'AMOE-master/AMOE-master/AMOE/images250/';

imgDir     = fullfile(baseDir, 'images');
slicDir    = fullfile(baseDir, 'slic');
g2bDir     = fullfile(baseDir, 'g2b');

fromslicDir = fullfile(fromDir, 'slic');
fromg2bDir  = fullfile(fromDir, 'g2b');

% Create output directories if they don't exist
if ~exist(slicDir, 'dir'), mkdir(slicDir); end
if ~exist(g2bDir, 'dir'), mkdir(g2bDir); end

% Get all files in imgDir, excluding directories
allFiles = dir(imgDir);
imgFiles = allFiles(~[allFiles.isdir]);  % Skip folders

% Start from index 21
startIdx = 21;
numImgs = length(imgFiles);

fprintf('Processing %d images (starting from index %d)...\n', numImgs - startIdx + 1, startIdx);

for i = startIdx:numImgs
    imgNameFull = imgFiles(i).name;
    [~, imgName, ~] = fileparts(imgNameFull);  % Remove extension

    fprintf('\n[%d/%d] Processing %s at %s...\n', i, numImgs, imgName, datestr(now));
    drawnow;
    pause(0.01);

    % Construct .mat file names
    slicMatFile = [imgName, '.mat'];
    g2bMatFile  = [imgName, '.mat'];

    % Source file paths
    fromSlicPath = fullfile(fromslicDir, slicMatFile);
    fromG2bPath  = fullfile(fromg2bDir,  g2bMatFile);

    % Destination file paths
    toSlicPath = fullfile(slicDir, slicMatFile);
    toG2bPath  = fullfile(g2bDir,  g2bMatFile);

    % Check and copy if both source files exist
    if exist(fromSlicPath, 'file') && exist(fromG2bPath, 'file')
        copyfile(fromSlicPath, toSlicPath);
        copyfile(fromG2bPath,  toG2bPath);
        fprintf('✔️  Copied %s.mat to slic and g2b directories.\n', imgName);
    else
        fprintf('⚠️  Skipping %s - .mat files not found in source directories.\n', imgName);
    end
end

fprintf('\n✅ Batch processing complete at %s.\n', datestr(now));
