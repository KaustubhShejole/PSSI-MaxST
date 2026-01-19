% Batch Processing for SSNCut Preprocessing (with gPb downsample + upsample)
baseDir = 'AMOE-master/AMOE-master/AMOE/imagesgrabcut/';
imgDir = fullfile(baseDir, 'images');
g2bDir = fullfile(baseDir, 'g2b');

if ~exist(g2bDir, 'dir'), mkdir(g2bDir); end

imgFiles = dir(fullfile(imgDir, '*.*'));
imgFiles = imgFiles(~[imgFiles.isdir]);  % Exclude folders

startIdx = 47;
numImgs = length(imgFiles);

fprintf('Processing %d images (starting from index %d)...\n', numImgs - startIdx + 1, startIdx);

for i = startIdx:numImgs
    imgNameFull = imgFiles(i).name;
    [~, imgName, ~] = fileparts(imgNameFull);
    fprintf('\n[%d/%d] Processing %s at %s...\n', i, numImgs, imgName, datestr(now));

    gPbPath = fullfile(g2bDir, [imgName, '.mat']);
    % if exist(gPbPath, 'file')
    %     fprintf('    ⏭️  Skipping — already exists.\n');
    %     continue;
    % end

    try
        %% Load original image
        imgFile = fullfile(imgDir, imgNameFull);
        img = imread(imgFile);
        originalSize = size(img);  % Store for later upsampling

        %% Downsample
        scale = 1.0;  % You can tweak this
        imgSmall = imresize(img, scale);

        %% Save downsampled temp image
        tmpImgPath = fullfile(tempdir, [imgName, '_tmp.jpg']);
        imwrite(imgSmall, tmpImgPath);

        %% Run globalPb on the downsampled image
        fprintf('    ⏳ Running globalPb on downsampled image...\n');
        gPb_orient = globalPb(tmpImgPath);  % Still slow, but faster than full-res

        %% Max across orientations and upsample back
        gPbSmall = max(gPb_orient, [], 3);
        gPb = imresize(gPbSmall, [originalSize(1), originalSize(2)], 'bilinear');

        %% Save result
        save(gPbPath, 'gPb');
        fprintf('    💾 Saved gPb for %s (resized to original size)\n', imgName);


        %% Compute SLIC
        cform = makecform('srgb2lab');
        imlab = applycform(img, cform);

        regionSize = 10;
        regularizer = 10000;

        spSegs = vl_slic(single(imlab), regionSize, regularizer);

        % Save Outputs
        save(fullfile(slicDir, [imgName, '.mat']), 'spSegs');

    catch ME
        fprintf('⚠️  Error with %s: %s\n', imgName, ME.message);
        continue;
    end
end

fprintf('\n✅ Batch complete at %s.\n', datestr(now));