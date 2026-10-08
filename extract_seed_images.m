% =========================================================
% Step 1a: Extract 300 random seed images for manual labelling
% =========================================================
% This script randomly selects 300 images from the full
% Indian_Digits_Train dataset and copies them into a separate
% folder called 'Seed_Images', ready for manual inspection.
%
% Run this script ONCE. After running, open 'Seed_Images' and
% label each image manually using create_seed_csv.m
% =========================================================

% Define source and destination folders
sourceFolder = 'Indian_Digits_Train';
destFolder   = 'Seed_Images';

% Create the destination folder if it doesn't already exist
if ~exist(destFolder, 'dir')
    mkdir(destFolder);
end

% Total images in the dataset and how many we want to seed
totalImages  = 10000;
numToSelect  = 300;

% Generate 300 unique random indices between 1 and 10,000
% randperm guarantees no repeats
rng(42);  % Fixed seed for reproducibility — same 300 images every run
randomIndices = randperm(totalImages, numToSelect);

% Save the selected indices to a .mat file so later scripts
% know exactly which image files belong to the seed set
save('seed_indices.mat', 'randomIndices');

disp('Starting to copy 300 random seed images...');

% Copy each selected image into the Seed_Images folder
for i = 1:numToSelect
    fileName   = sprintf('%d.bmp', randomIndices(i));
    sourcePath = fullfile(sourceFolder, fileName);
    destPath   = fullfile(destFolder,   fileName);
    copyfile(sourcePath, destPath, 'f');
end

fprintf('Done! 300 images copied to "%s".\n', destFolder);
disp('Next step: run create_seed_csv.m to generate the labelling spreadsheet.');
