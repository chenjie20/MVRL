function [dynamical_set, truncated_index_set] = update_dynamical_set(data_views, Zn, cluster_lables, window_size)

    nv = length(data_views);
    dynamical_set = cell(1, nv);
    num_clusters = length(unique(cluster_lables));
    max_num_per_cluster = floor(window_size / num_clusters);
    data_view_cache = cell(1, nv);
    for nv_idx = 1 : nv
        data_view_cache{nv_idx} = zeros(size(data_views{nv_idx}, 1), max_num_per_cluster * num_clusters);    
    end
    original_index_set = zeros(1,  max_num_per_cluster * num_clusters);
    
    num_selected_samples =  0;
    for cluster_idx = 1 : num_clusters
        index_set = cluster_lables == cluster_idx;
        current_index_set = find(index_set > 0);
        len_samples = length(current_index_set);
        reconstruction_residuals = zeros(1, len_samples);
        for idx = 1 : len_samples
            for nv_idx = 1 : nv
                data_view_subset = normc(data_views{nv_idx}(:, index_set));
                Zn_subset = Zn{nv_idx}(index_set, index_set);
                data_view_subset(:, idx) = [];
                Zn_subset(:, idx) = [];
                Zn_subset(idx, :) = [];
                reconstruction_residuals(1, idx) = reconstruction_residuals(1, idx) + norm((data_view_subset - data_view_subset * Zn_subset), 'fro');
            end
        end
        if len_samples > max_num_per_cluster
           [~, new_col_positions] = sort(reconstruction_residuals, 'descend');
           selected_cols = new_col_positions(1 : max_num_per_cluster);
           current_index_set_in_data_views = current_index_set(selected_cols);
        else
            selected_cols = 1 : len_samples; 
            current_index_set_in_data_views = current_index_set;
        end        
        start_index = num_selected_samples + 1;
        num_selected_samples = num_selected_samples + length(selected_cols); 
        for nv_idx = 1 : nv
            data_view_subset = data_views{nv_idx}(:, index_set);
            data_view_cache{nv_idx}(:, start_index : num_selected_samples) = data_view_subset(:, selected_cols);        
        end
        original_index_set(1, start_index : num_selected_samples) = current_index_set_in_data_views;
    end
    for nv_idx = 1 : nv
        dynamical_set{nv_idx} = data_view_cache{nv_idx}(: , 1 : num_selected_samples);
    end
    truncated_index_set = original_index_set(1, 1 : num_selected_samples);
    
end
