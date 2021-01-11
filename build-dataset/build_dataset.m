%% Generate a dataset for the Zynq-7000 SoC KNN
% Janus Bo Andersen, Nov/Dec 2020.

clear; clc; close all;

%%
% Load the M&Ms dataset instead of rebuilding it

%load('mm_dataset.mat');

%% Rebuild M&Ms dataset manually
% make dataStore of images
imds = imageDatastore('img/');
imgs = readall(imds);

%%
% Manually annotated colours for the 32 images in same order as images
label_names = {'blue','yellow','brown','red','blue','brown', 'yellow', ...
    'brown', 'red', 'yellow', 'blue', 'brown', 'yellow', 'red', 'green',...
    'red', 'blue', 'green', 'red', 'yellow', 'red', 'brown', 'red', ...
    'red', 'brown', 'blue',  'brown', 'brown', 'green', 'brown', 'brown',...
    'yellow'};

%%
% Hashmap for the M&Ms to translate from name to id, e.g. 'blue' -> 1
% The 5 M&Ms colours and associated colour ids
key_set = {'blue', 'yellow', 'brown', 'red', 'green'};
id_set = [1,2,3,4,5];
M = containers.Map(key_set, id_set);

%%
% Transform the names to ids
label_ids = {};
for k = 1:size(label_names,2)
    label_ids{k} = M(label_names{k});
end

%%
% Manually crop the first 32 images and store label together

for k = 1:32
    [J, rect] = imcrop(imgs{k});    % manually crop kth image
    mm_dataset{k}.im = J;
    mm_dataset{k}.rect = rect;
    mm_dataset{k}.label_id = label_ids{k};
    mm_dataset{k}.label_name = label_names{k};
end

%%
% Augment the dataset with labels on top of the images for visualization
imgs_for_tile = {};

for k = 1:size(mm_dataset,2)
    mm_dataset{k}.im_label = insertText(mm_dataset{k}.im, [10 60], ...
        mm_dataset{k}.label_name, 'AnchorPoint', 'LeftBottom', ...
        'BoxColor', 'w', 'FontSize', 32);
    imgs_for_tile{k} = mm_dataset{k}.im_label;
end


%%
% Visualize the whole dataset with labels
tiled = imtile(imgs_for_tile);
figure; imshow(tiled)
title('Manually labelled dataset of M&Ms', 'FontSize', 16);

%%
% Save dataset
% save('mm_dataset.mat', 'mm_dataset');

%% Sampling of feature vectors from dataset
% Sample the dataset using an isotropic Gaussian distribution, 
% centered around the midpoint of the image.
% 
% This will create some noise when samples are taken outside the M&M, but
% that is okay, it doesn't need to be perfect.
% 
% We want feature vectors of about (64x1) to get a very robust classifier
% but using RGB gives three values per image pixel, so we can sample 
% 21 pixels and build a feature vector of dimension (63x1).
% 
% If we sample the same image 32 times (at different random locations), 
% we then get a total of 32*32 = 1024 labelled reference vectors.

rng(42);                    % seed for reproducability
num_points = 21;            % pixels sampled per image
rgb = 3;                    % 3-vector per pixel
num_samp_per_im = 32;       % sample each im. 32 times
num_im = 32;                % there are 32 im.s in the set

% Reference dataset variables
X = zeros([num_im*num_samp_per_im, num_points*rgb]);    % 1024 feature vecs
y_name = {};                                            % Label strings
y_id = zeros([num_im*num_samp_per_im, 1]);              % Label ids

z_frac = 1.5;          % Gaussian quantile for the sampling distribution,
                        % the lower the number, the more noise.
                        % 2.58 -> about 1% misses,
                        % 1.96 -> about 5% misses,
                        % 1.64 -> about 10% misses,
                        % 1.50 -> about 12-14% misses, etc...
                        % E.g. norminv([0.005 .995])-> [-/+ 2.58]


sample_idx = 1;     % Running counter

% BEGIN SAMPLING
for im_n = 1:num_im

    % Select image and compute settings
    samp_im = mm_dataset{im_n}.im;
    s = size(samp_im);

    % Central and max sampling coordinates
    y_ctr = s(1) / 2; y_max = s(1);
    x_ctr = s(2) / 2; x_max = s(2);

    % Sample the same image multiple times at different locations
    for sam_n = 1:num_samp_per_im
        
        % Sample locations for x-coordinate
        nrx = round(x_ctr + (x_ctr/z_frac)*randn([num_points, 1]));            
        nrx(nrx <= 0) = 1;   % cap at min
        nrx(nrx > x_max) = x_max; % cap at max

        % Sample locations for y-coordinate
        nry = round(y_ctr + (y_ctr/z_frac)*randn([num_points, 1]));
        nry(nry <= 0) = 1;
        nry(nry > y_max) = y_max;
        
        % Obtain samples
        samples = zeros([num_points 3]); % Sample vector x 3 for RGB

        % sample the given number of different points in this image for
        % this feature vector
        for k = 1:num_points
            samples(k, :) = samp_im(nry(k), nrx(k), :);  % RGB from im(x,y)
        end % end of sampling 21 points in same image for one feature vec

        % Organize so 21*R, 21*G, and 21*B
        feature_vec = reshape(samples, [], 1); 
        
        % Store feature vector and labels
        X(sample_idx, :) = feature_vec';   % store feature vec
        y_id(sample_idx, 1) = mm_dataset{im_n}.label_id;
        y_name{end + 1} = mm_dataset{im_n}.label_name;
        
        % increment counter
        sample_idx = sample_idx + 1;

    end % end of sampling same image 32 times
end % end of sampling all images


%%
% Display last sampling points for demo
figure; imshow(samp_im); hold on;
plot(nrx, nry, 'r+');
hold off;
title('Bivariate Gaussian sampling points for an image');


%%
% Build query set (test set with labels, the labels are for accuracy test)
% Randomly select among the 32 images and perform a random sampling of the
% image until 100 images have been sampled.
% Store the labels to do accuracy measurement (backtest)

num_q_vecs = 100;   % 100 query vectors
s_ims = randi([1 num_im], num_q_vecs, 1);    % Sampled image ids

Xq = zeros([num_q_vecs, num_points*rgb]);    % Feature vec's for query
yq_name = {};                                % Label strings
yq_id = zeros([num_q_vecs, 1]);              % Label ids


% Actually build the query set
rng(4242);

sample_idx = 1;     % Running counter

for im_n = 1:num_q_vecs

    % Select random image and compute settings
    samp_im_d = mm_dataset{s_ims(im_n)};
    samp_im = samp_im_d.im;   % get a random im
    s = size(samp_im);

    % Central and max sampling coordinates
    y_ctr = s(1) / 2; y_max = s(1);
    x_ctr = s(2) / 2; x_max = s(2);

    % For now, just do one sample per image
    for sam_n = 1:1
        
        % Sample locations for x-coordinate
        nrx = round(x_ctr + (x_ctr/z_frac)*randn([num_points, 1]));            
        nrx(nrx <= 0) = 1;   % cap at min
        nrx(nrx > x_max) = x_max; % cap at max

        % Sample locations for y-coordinate
        nry = round(y_ctr + (y_ctr/z_frac)*randn([num_points, 1]));
        nry(nry <= 0) = 1;
        nry(nry > y_max) = y_max;
        
        % Obtain samples
        samples = zeros([num_points 3]); % Sample vector x 3 for RGB

        % sample the given number of different points in this image for
        % this feature vector
        for k = 1:num_points
            samples(k, :) = samp_im(nry(k), nrx(k), :);  % RGB from im(x,y)
        end % end of sampling 21 points in same image for one feature vec

        % Organize so 21*R, 21*G, and 21*B
        feature_vec = reshape(samples, [], 1); 
        
        % Store feature vector and labels
        Xq(sample_idx, :) = feature_vec';   % store feature vec
        yq_id(sample_idx, 1) = samp_im_d.label_id;
        yq_name{end + 1} = samp_im_d.label_name;
        
        % increment counter
        sample_idx = sample_idx + 1;

    end % end of sampling same image 32 times
end % end of sampling all images

%%
% Display last sampling points for demo
figure; imshow(samp_im); hold on;
plot(nrx, nry, 'r+');
hold off;
title('Bivariate Gaussian sampling points for an image');

%% Export the dataset to text files
% This is importable as a variable in a C header file
% In C, we load the reference (training) set as an array of 1024*63 ints,
% and we load the query (test) set as an array of 100*63 = 6300 ints.
%
% This is:
% - 258.0 kilobyte for the reference data, and 
% - 25.2 kilobyte for the training set.

% First the reference/training data
data = X';      % to get the order I want! Full row, then next row, then...
len_data = size(data, 1)*size(data, 2);
fid_data = fopen('ref_vals.txt', 'wt');

% Write data to file as 1, 2, 3, 4, ...
for k = 1:len_data
    fprintf(fid_data, '%d', data(k));  % write value
    
    if k < len_data
        fprintf(fid_data, ',\n');        % write the comma
    end
end

fclose(fid_data);

% Then the corresponding labels (colour ids)
fid_labels = fopen('ref_labels.txt', 'wt');
labels = y_id;
len_labels = size(labels, 1)*size(labels, 2);
for k = 1:len_labels
    fprintf(fid_labels, '%d', labels(k));
    if k < len_labels
        fprintf(fid_labels, ',\n');
    end
end
fclose(fid_labels);

% The query/test data
data = Xq';      % to get the order I want! Full row, then next row, then...
len_data = size(data, 1)*size(data, 2);
fid_data = fopen('query_vals.txt', 'wt');
for k = 1:len_data
    fprintf(fid_data, '%d', data(k));
    if k < len_data
        fprintf(fid_data, ',\n');
    end
end
fclose(fid_data);

% Then the corresponding labels (colour ids)
fid_labels = fopen('query_labels.txt', 'wt');
labels = yq_id;
len_labels = size(labels, 1)*size(labels, 2);
for k = 1:len_labels
    fprintf(fid_labels, '%d', labels(k));
    if k < len_labels
        fprintf(fid_labels, ',\n');
    end
end
fclose(fid_labels);


%% Algorithm development
% Develop KNN for K=1 

N = size(X, 1);     % Reference vectors in the dataset
fd = size(X, 2);    % Feature dimensions (63)
Nqv = size(Xq, 1);  % Query vectors in the dataset

% Find distance between one query vector and all N reference vectors
q = 1;          % look at query vector #1
qv = Xq(q, :);

% Compute distance to reference vector 1
rv = X(1, :);
sum_dist = 0;
for d = 1:fd
    sum_dist = sum_dist + (qv(d) - rv(d))^2;    % Sq. Euclidian dist
end

% Test that it gives same as squared norm of difference
assert (sum_dist == norm(qv-rv)^2)

% For all reference vectors -> vector of sq. Euclidian distances
% Nested loop
best_dist = Inf;
nearest_neigbour = 0;
for vecnum = 1:N
    sum_dist = 0;   % reset
    for d = 1:fd
        sum_dist = sum_dist + (qv(d) - X(vecnum, d))^2;
    end
    if (sum_dist < best_dist)       % New nearest neighbour
        best_dist = sum_dist;       % Store the sq-dist to nearest
        nearest_neigbour = vecnum;  % Store the nearest so far
    end
end

% Find colour of nearest neighbour
predicted_colour = y_name(nearest_neigbour);
disp(['Nearest neighbour is ', predicted_colour{1}]);

% Check
actual_colour = yq_name(q);
disp(['The query vector is ', actual_colour{1}]);


%% Test dataset using Matlab's KNN
% Test this dataset using Matlab's KNN, with K = 1

Mdl = fitcknn(X,y_name,'NumNeighbors', 1);

predictions = Mdl.predict(Xq);  % Predict the query data based on training
actual = yq_name';              % the labels we saved from random samples

correct = 0;
for k=1:num_q_vecs
    if strcmp(predictions{k}, actual{k})
        correct = correct + 1;
    end
end

accuracy = correct / num_q_vecs;
disp(['Accuracy: ', num2str(round(accuracy * 100, 2)), '%.']);

%% Test own implementation and compare to Matlab
% Run for all 100 test vectors
own_knn1_predict = predict_knn1(X, y_name, Xq)';

% compare to Matlabs predictions
Mdl = fitcknn(X,y_name,'NumNeighbors', 1);
matlab_knn1_predict = Mdl.predict(Xq);

for k = 1:100
    assert (strcmp(own_knn1_predict{k}, matlab_knn1_predict{k}) == true)
end

% ALL GOOD!

% find the misses for our own
for k = 1:100
    own_knn1_misses(k) = ~strcmp(own_knn1_predict{k}, yq_name{k});
end

own_knn1_accuracy = 1 - sum(own_knn1_misses) / 100;


%% Algorithm development
% Develop KNN for any K

K = 3;

q = 1;                          % look at query vector #1
qv = Xq(q, :);

d2 = zeros([N 1]);              % Store sq. Euclidian dists
for vecnum = 1:N
    sum_dist = 0;               % reset sum
    for d = 1:fd
        sum_dist = sum_dist + (qv(d) - X(vecnum, d))^2;
    end
    d2(vecnum) = sum_dist;      % Store the distance
end


% Find K nearest neighbours by sorting and searching
sort_dists = sort(d2, 'ascend');
K_min_dists = sort_dists(1:K);
for k = 1:K
    K_min_idx(k) = find(d2 == K_min_dists(k));
    K_min_clr_ids(k) = y_id(K_min_idx(k));
end

% Voting
knn = mode(K_min_clr_ids);          % This might be a bit tricky to impl.

% Translate from id to colour name, e.g. 1 -> 'blue'
clr_names = {'blue', 'yellow', 'brown', 'red', 'green'};

% Find colour of nearest neighbour
predicted_colour = clr_names(knn);
disp(['Nearest neighbour classification (K=', num2str(K),') is ', ...
    predicted_colour{1}]);

% Check
actual_colour = yq_name(q);
disp(['The query vector is ', actual_colour{1}]);

%% Test against Matlab
%

own_knn3_predict = predict_knnK(X, y_id, Xq, 3);

% find the misses for our own
for k = 1:100
    own_knn3_misses(k) = ~strcmp(own_knn3_predict{k}, yq_name{k});
end

own_knn3_accuracy = 1 - sum(own_knn3_misses) / 100;

% Train matlab model
Mdl = fitcknn(X,y_name,'NumNeighbors', 3);
matlab_knn3_predict = Mdl.predict(Xq);

% find the misses for Matlab
for k = 1:100
    matlab_knn3_misses(k) = ~strcmp(matlab_knn3_predict{k}, yq_name{k});
end

% Compare predictions
for k = 1:100
    assert (strcmp(own_knn3_predict{k}, matlab_knn3_predict{k}) == true)
end
%%
% 

function [predictions] = predict_knn1(X, y_name, Xq)
% Make predictions using KNN with K = 1
% Janus Bo Andersen, December 2020

N = size(X, 1);     % Reference vectors in the dataset
fd = size(X, 2);    % Feature dimensions (63)
Nqv = size(Xq, 1);  % Query vectors in the dataset

predictions = {};

% Loop over all query vectors
for q = 1:Nqv
    qv = Xq(q, :);

    % For all reference vectors -> vector of sq. Euclidian distances
    % Nested loop
    best_dist = Inf;
    nearest_neigbour = 0;
    for vecnum = 1:N
        sum_dist = 0;   % reset
        for d = 1:fd    % loop over all features
            sum_dist = sum_dist + (qv(d) - X(vecnum, d))^2;
        end
        if (sum_dist < best_dist)       % New nearest neighbour
            best_dist = sum_dist;       % Store the sq-dist to nearest
            nearest_neigbour = vecnum;  % Store the nearest so far
        end
    end

    % Find colour of nearest neighbour
    predicted_label = y_name(nearest_neigbour); % KNN K = 1, easy!
    predictions{q} = predicted_label{1};        % Remove cell wrapper

end % end of loop over all q

end

%%
%
function [predictions] = predict_knnK(X, y_id, Xq, K)
% Make predictions using our own KNN model, for any choice of K
% Janus Bo Andersen, December 2020

% Translate from id to colour name, e.g. 1 -> 'blue'
clr_names = {'blue', 'yellow', 'brown', 'red', 'green'};

N = size(X, 1);     % Reference vectors in the dataset
fd = size(X, 2);    % Feature dimensions (63)
Nqv = size(Xq, 1);  % Query vectors in the dataset

predictions = {};

for q = 1:Nqv
    qv = Xq(q, :);                  % Choose query vector

    d2 = zeros([N 1]);              % Store sq. Euclidian dists
    for vecnum = 1:N
        sum_dist = 0;               % reset sum
        for d = 1:fd
            sum_dist = sum_dist + (qv(d) - X(vecnum, d))^2;
        end
        d2(vecnum) = sum_dist;      % Store the distance
    end

    % Find K nearest neighbours by sorting and searching
    sort_dists = sort(d2, 'ascend');
    K_min_dists = sort_dists(1:K);
    for k = 1:K
        K_min_idx(k) = find(d2 == K_min_dists(k));
        K_min_clr_ids(k) = y_id(K_min_idx(k));
    end

    % Voting among the K nearest neighbours
    knn = mode(K_min_clr_ids);  % This might be tricky to impl. in C++
    
    % Find colour of nearest neighbour
    predicted_colour = clr_names(knn);
    predictions{q} = predicted_colour{1};

end % end for q

end