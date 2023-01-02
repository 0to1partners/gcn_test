

args = {
    # Fixed
    'epochs': 100,  # 에폭 수
    'batch_size': 1,
    'lr' : 0.001,

    # Hyperparameter
    # Adjacency Matrix Hyperparameter
    'adj_hidden_dim': 8,
    'adj_num_layers': 2,
    'adj_embedding_dim': 2,
    'adj_channel': 1,
    'adj_input_dim': 20,
    'adj_output_dim': 2,

    # Graph Module Hyperparameter
    'node_dim': 28,
    'node_cnt': 25,
    'node_latent_dim': 16,
    'node_hidden_dim': 32,
    'node_num_layers': 3,

    # Train & Predict Sequence Length
    'seq_len': 24,
    'pred_len': 1,

    # Auxiliary Data Information
    'temporal_embedding_dim' : 4,
    'temporal_columns' : [
        'day',
        'holiday',
        'temp',
        ],
    'temporal_cardinalities' : [7,2,1],

    'spatial_embedding_dim' : 4,
    'spatial_columns' : [
        'cnt_worker_male_2019',
        'cnt_worker_female_2019',
        'culture_cnt_2020',
        'physical_facil_2019',
        'school_cnt_2020', 
        'student_cnt_2020',
        ],
    'spatial_cardinalities' : [1,1,1,1,1,1],
}
