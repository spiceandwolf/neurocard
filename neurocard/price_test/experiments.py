"""Experiment configurations.

EXPERIMENT_CONFIGS holds all registered experiments.

TEST_CONFIGS (defined at end of file) stores "unit tests": these are meant to
run for a short amount of time and to assert metrics are reasonable.

Experiments registered here can be launched using:

  >> python run.py --run <config> [ <more configs> ]
  >> python run.py  # Runs all tests in TEST_CONFIGS.
"""
import os

from ray import tune

EXPERIMENT_CONFIGS = {}
TEST_CONFIGS = {}

# Common config. Each key is auto set as an attribute (i.e. NeuroCard.<attr>)
# so try to avoid any name conflicts with members of that class.
BASE_CONFIG = {
    'cwd': os.getcwd(),
    'epochs_per_iteration': 1,
    'num_eval_queries_per_iteration': 100,
    'num_eval_queries_at_end': 2000,  # End of training.
    'num_eval_queries_at_checkpoint_load': 2000,  # Evaluate a loaded ckpt.
    'epochs': 10,
    'seed': None,
    'order_seed': None,
    'bs': 2048,
    'order': None,
    'layers': 2,
    'fc_hiddens': 128,
    'warmups': 1000,
    'constant_lr': None,
    'lr_scheduler': None,
    'custom_lr_lambda': None,
    'optimizer': 'adam',
    'residual': True,
    'direct_io': True,
    'input_encoding': 'embed',
    'output_encoding': 'embed',
    'query_filters': [5, 12],
    'force_query_cols': None,
    'embs_tied': True,
    'embed_size': 32,
    'input_no_emb_if_leq': True,
    'resmade_drop_prob': 0.,

    # Multi-gpu data parallel training.
    'use_data_parallel': False,

    # If set, load this checkpoint and run eval immediately. No training. Can
    # be glob patterns.
    # Example:
    # 'checkpoint_to_load': tune.grid_search([
    #     'models/*52.006*',
    #     'models/*43.590*',
    #     'models/*42.251*',
    #     'models/*41.049*',
    # ]),
    'checkpoint_to_load': None,
    # Dropout for wildcard skipping.
    'disable_learnable_unk': False,
    'per_row_dropout': True,
    'dropout': 1,
    'table_dropout': False,
    'fixed_dropout_ratio': False,
    'asserts': None,
    'special_orders': 0,
    'special_order_seed': 0,
    'join_tables': [],
    'label_smoothing': 0.0,
    'compute_test_loss': False,

    # Column factorization.
    'factorize': False,
    'factorize_blacklist': None,
    'grouped_dropout': True,
    'factorize_fanouts': False,

    # Eval.
    'eval_psamples': [100, 1000, 10000],
    'eval_join_sampling': None,  # None, or #samples/query.

    # Transformer.
    'use_transformer': False,
    'transformer_args': {},

    # Checkpoint.
    'save_checkpoint_at_end': True,
    'checkpoint_every_epoch': False,

    # Experimental.
    '_save_samples': None,
    '_load_samples': None,
    'num_orderings': 1,
    'num_dmol': 0,
}

TEST_DATASET_BASE = {
    'dataset': '',
    'join_tables': [],
    'join_keys': {},
    # Sampling starts at this table and traverses downwards in the join tree.
    'join_root': '',
    # Inferred.
    'join_clauses': None,
    'join_how': 'outer',
    # Used for caching metadata.  Each join graph should have a unique name.
    'join_name': '',
    'seed': 0,
    'per_row_dropout': False,
    'table_dropout': True,
    'embs_tied': True,
    # Num tuples trained =
    #   bs (batch size) * max_steps (# batches per "epoch") * epochs.
    'epochs': 1,
    'bs': 2048,
    'max_steps': 500,
    # Use this fraction of total steps as warmups.
    'warmups': 0.05,
    # Number of DataLoader workers that perform join sampling.
    'loader_workers': 8,
    # Options: factorized_sampler, fair_sampler (deprecated).
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1024 * 4,
    'layers': 4,
    # Eval:
    'compute_test_loss': True,
    'queries_csv': '/home/user/oblab/CE-baselines/test_dataset_training/neurocard',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 3000,
    'eval_psamples': [4000],

    # Multi-order.
    'special_orders': 0,
    'order_content_only': True,
    'order_indicators_at_front': False,
    
    # test_datasets
    'test_datasets': ['accidents', 'carcinogenesis', 'consumer', 'hockey', 'ssb', 'talkingData'],
    'datasets_path': '/home/user/oblab/PRICE/datas/datasets/',
}

FACTORIZE = {
    'factorize': True,
    'word_size_bits': 10,
    'grouped_dropout': True,
}


### EXPERIMENT CONFIGS ###
# Run multiple experiments concurrently by using the --run flag, ex:
# $ ./run.py --run job-light
EXPERIMENT_CONFIGS = {
    'test': dict(
        dict(BASE_CONFIG, **TEST_DATASET_BASE),
        **{
            'factorize': True,
            'word_size_bits': 10,
            'grouped_dropout': True,
            'loader_workers': 4,
            'warmups': 0.05,  # Ignored.
            'lr_scheduler': tune.grid_search(['OneCycleLR-0.28']),
            'loader_workers': 4,
            'max_steps': tune.grid_search([500]),
            'epochs': 7,
            'num_eval_queries_per_iteration': 70,
            'input_no_emb_if_leq': False,
            'eval_psamples': [8000],
            'epochs_per_iteration': 1,
            # 'resmade_drop_prob': tune.grid_search([.1]),
            # 'label_smoothing': tune.grid_search([0]),
            # 'word_size_bits': tune.grid_search([11]),
            
        }),
}



######  TEST CONFIGS ######
# These are run by default if you don't specify --run.

TEST_CONFIGS['test-datasets'] = dict(
    EXPERIMENT_CONFIGS['test'],
    **{
        # Train for a bit and checks that these metrics are reasonable.
        'epochs': 1,
        'asserts': {
            'fact_psample_8000_median': 4,
            'fact_psample_8000_p99': 50,
            'train_bits': 80,
        },
    })

for name in TEST_CONFIGS:
    TEST_CONFIGS[name].update({'save_checkpoint_at_end': False})
EXPERIMENT_CONFIGS.update(TEST_CONFIGS)
