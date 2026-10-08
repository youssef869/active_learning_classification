% =========================================================
% Step 1b: Create a CSV spreadsheet for manual seed labelling
% =========================================================
% Run this script AFTER extract_seed_images.m
% It reads all images in 'Seed_Images', creates a CSV with
% one row per image and an empty 'Label' column.
% =========================================================

% Folder containing the 300 copied seed images
folderPath = 'Seed_Images';

% Get all .bmp files in the folder
imageFiles = dir(fullfile(folderPath, '*.bmp'));
numFiles   = length(imageFiles);

% Sort filenames numerically (dir returns alphabetical order
% which sorts '10.bmp' before '2.bmp', so we fix that here)
fileNames = {imageFiles.name};
fileNums  = cellfun(@(f) sscanf(f, '%d.bmp'), fileNames);
[~, sortIdx] = sort(fileNums);
imageFiles   = imageFiles(sortIdx);

% Build the CSV cell array with a header row
csvData        = cell(numFiles + 1, 2);
csvData{1, 1}  = 'Filename';
csvData{1, 2}  = 'Label';

% Fill in filenames — Label column left blank for you to fill
for i = 1:numFiles
    csvData{i + 1, 1} = imageFiles(i).name;
    csvData{i + 1, 2} = '';   % <-- you fill this in manually
end

% Write to CSV
outputFile = 'seed_labels.csv';
writecell(csvData, outputFile);

fprintf('Spreadsheet created: "%s"\n', outputFile);
fprintf('It contains %d rows (one per image).\n', numFiles);
disp('Open it, fill in the Label column (0-9), save, then proceed to Step 2.');
