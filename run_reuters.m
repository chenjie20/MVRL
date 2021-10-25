close all;
clear;
clc;

addpath('data');
addpath('utility');

% load the training data from the MNIST database files.
load('reuters.mat');

n = 500;
total_num = length(Y);
% num_windows = floor(total_num / n);
num_windows = 30;
Y = Y + 1;
K = max(Y);
max_num_per_cluster = floor(n / K);
nv = size(X, 2);

rand('state', 100);
rand_set = randperm(total_num);
rand_data_views = cell(1, nv);
for nv_idx = 1 : nv 
    rand_data_views{nv_idx} = X{nv_idx}(rand_set, :)';    
end
gnd =  Y(rand_set, 1)';

alphas = [0.05];
dims = [10]; 

alpha_num = length(alphas);
dim_num = length(dims);

final_clustering_accs = zeros(alpha_num, dim_num, num_windows);
final_clustering_nmis = zeros(alpha_num, dim_num, num_windows);
final_clustering_purities = zeros(alpha_num, dim_num, num_windows);
final_clustering_fmeasures = zeros(alpha_num, dim_num, num_windows);
final_clustering_costs = zeros(alpha_num, dim_num, num_windows);
final_clustering_ratios =  zeros(alpha_num, dim_num, num_windows);

original_data_views = cell(1, nv);
data_views = cell(1, nv);
data_views_subset = cell(1, nv);
Zn_subset = cell(1, nv);
Xs = cell(1, nv);
for alpha_idx = 1 : alpha_num
    alpha = alphas(alpha_idx);     
    for dim_idx = 1 : length(dims)
        dim = dims(dim_idx);
        stream = RandStream.getGlobalStream;
        reset(stream);
        enable_k = 1; % The number of clusters is automatically determined.  
        for wnd_idx = 1 : num_windows
            disp(wnd_idx);            
            % 1. Get data and labels
            start_idx = (wnd_idx - 1) * n + 1;
            for nv_idx = 1 : nv                                 
                if wnd_idx == 1
                    original_data_views{nv_idx} = rand_data_views{nv_idx}(:, start_idx : start_idx + n - 1);

                else
                    original_data_views{nv_idx} = [rand_data_views{nv_idx}(:, start_idx : start_idx + n - 1), Xs{nv_idx}(:, :)];
                end
            end
            ground_lables = gnd(start_idx : start_idx + n - 1); 

            % 2. Clustering
            %data preprocessing
            for nv_idx = 1 : nv
                if dim > 0 && dim < size(original_data_views{nv_idx}, 1)
                    [eigen_vector, ~, ~] = f_pca(original_data_views{nv_idx}, dim);
                    data_views{nv_idx} = eigen_vector' * original_data_views{nv_idx};
                else
                    data_views{nv_idx} = original_data_views{nv_idx};
                end
            end
            tic;
            [W, Zn] = mvrl(data_views, alpha);
            time_cost = toc;
            len_W = size(W, 2);
            ratio = length(find(abs(W) > 1e-6)) / (len_W * len_W);                                    
            [actual_ids, num_sc_clusters] = spectral_clustering_with_max_k(W, K, enable_k);                    
            if num_sc_clusters == K
                enable_k = 0;
            end
            [current_ground_lables, ~] = refresh_labels(ground_lables, K);
            [current_cluster_lables, ~] = refresh_labels(actual_ids(1 : n, 1)', K);
            num_current_clusters = length(unique(current_cluster_lables));                    
            [acc, nmi, purity, fmeasure, ~, ~] = calculate_dynamic_clustering_results(current_cluster_lables, current_ground_lables, num_current_clusters);
            final_clustering_accs(alpha_idx, dim_idx, wnd_idx) = acc;
            final_clustering_nmis(alpha_idx, dim_idx, wnd_idx) = nmi;
            final_clustering_purities(alpha_idx, dim_idx, wnd_idx) = purity;
            final_clustering_fmeasures(alpha_idx, dim_idx, wnd_idx) = fmeasure;
            final_clustering_ratios(alpha_idx, dim_idx, wnd_idx) = ratio;

            % 3. Adaptively update the representative data objects in a dynamical set
            [dynamical_set, original_index_set] = update_dynamical_set(data_views, Zn, actual_ids, n);  
            for nv_idx = 1 : nv
                Xs{nv_idx} = original_data_views{nv_idx}(:, original_index_set);
            end                                
            final_clustering_costs(alpha_idx, dim_idx, wnd_idx) = time_cost;
            disp([wnd_idx, alpha, dim, acc, nmi, purity, fmeasure, ratio, time_cost]);            
            dlmwrite('mvrl_reuter_data_parameters_new.txt', [wnd_idx, alpha, dim, acc, nmi, purity, fmeasure, ratio, time_cost] , '-append', 'delimiter', '\t', 'newline', 'pc');   
        end
        average_acc =  mean(final_clustering_accs(alpha_idx, dim_idx, :));
        average_nmi =  mean(final_clustering_nmis(alpha_idx, dim_idx, :));
        average_purity =  mean(final_clustering_purities(alpha_idx, dim_idx, :));            
        average_fm =  mean(final_clustering_fmeasures(alpha_idx, dim_idx, :));            
        average_cost = mean(final_clustering_costs(alpha_idx, dim_idx, :));   
        average_ratio =  mean(final_clustering_ratios(alpha_idx, dim_idx, :));    
        disp([alpha, dim, average_acc, average_nmi, average_purity, average_fm, average_ratio, average_cost]);
        dlmwrite('mvrl_reuter_avg_results_parameters_new.txt', [alpha, dim, average_acc, average_nmi, average_purity, average_fm, average_ratio, average_cost] , '-append', 'delimiter', '\t', 'newline', 'pc');
    end
end
save('mvrl_reuter_results_final_parameters.mat', 'final_clustering_accs', 'final_clustering_nmis', 'final_clustering_purities', 'final_clustering_fmeasures', 'final_clustering_ratios', 'final_clustering_costs');
