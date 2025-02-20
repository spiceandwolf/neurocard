"""Registry of datasets and schemas."""
import collections
import os
import pickle

import numpy as np
import pandas as pd

import collections

import common


datasets_path = f'/opt/hdd/datasets/user/datasets/'

def CachedReadCsv(filepath, **kwargs):
    """Wrapper around pd.read_csv(); accepts same arguments."""
    parsed_path = filepath[:-4] + '.df'
    if os.path.exists(parsed_path):
        with open(parsed_path, 'rb') as f:
            df = pickle.load(f)
        assert isinstance(df, pd.DataFrame), type(df)
        print('Loaded parsed csv from', parsed_path)
    else:
        df = pd.read_csv(filepath, **kwargs)
        with open(parsed_path, 'wb') as f:
            # Use protocol=4 since we expect df >= 4GB.
            pickle.dump(df, f, protocol=4)
        print('Saved parsed csv to', parsed_path)
    return df


class TestDataset(object):
    ALIAS_TO_TABLE_NAME = {}

    # Columns where only equality filters make sense.
    CATEGORICAL_COLUMNS = collections.defaultdict(list)

    # Columns with a reasonable range/IN interpretation.
    RANGE_COLUMNS = collections.defaultdict(list)

    CSV_FILES = []
    
    TEST_DATASET_PRED_COLS = collections.defaultdict(list)
    
    tbls_cols_types = {}

    # For testdataset schema.
    TRUE_FULL_OUTER_CARDINALITY = {
        ('customer', 'dim_date', 'lineorder', 'part', 'supplier'): 12218146,
    }

    # CSV -> RANGE union CATEGORICAL columns.
    _CONTENT_COLS = None
    
    def __init__(self, config):
        TestDataset.ALIAS_TO_TABLE_NAME = config['ALIAS_TO_TABLE_NAME']
        TestDataset.CATEGORICAL_COLUMNS = config['CATEGORICAL_COLUMNS']
        TestDataset.RANGE_COLUMNS = config['RANGE_COLUMNS']
        TestDataset.CSV_FILES = config['CSV_FILES']
        TestDataset.TEST_DATASET_PRED_COLS = config['TEST_DATASET_PRED_COLS']
        TestDataset.tbls_cols_types = config['tbls_cols_types']
        
    @staticmethod
    def ContentColumns():
        if TestDataset._CONTENT_COLS is None:
            TestDataset._CONTENT_COLS = {
                '{}.csv'.format(table_name):
                range_cols + TestDataset.CATEGORICAL_COLUMNS[table_name]
                for table_name, range_cols in
                TestDataset.RANGE_COLUMNS.items()
            }
            # Add join keys.
            for table_name in TestDataset._CONTENT_COLS:
                cols = TestDataset._CONTENT_COLS[table_name]
                if table_name == 'lineorder.csv':
                    cols.append('lo_suppkey')
                # elif 'movie_id' in JoinOrderBenchmark.BASE_TABLE_PRED_COLS[
                #         table_name]:
                #     cols.append('movie_id')

        return TestDataset._CONTENT_COLS

    @staticmethod
    def GetFullOuterCardinalityOrFail(join_tables):
        key = tuple(sorted(join_tables))
        return TestDataset.TRUE_FULL_OUTER_CARDINALITY[key]

    def GetTestDatasetJoinKeys(file_fanout_path):
        """the file_fanout_path is always in the path of '/the source code of PRICE/datas/statistics/{args.usage}/{args.db}/gen_fanout{args.bs}.pkl'"""
        with open(file_fanout_path, 'rb') as f:
            fanout = pickle.load(f)
        joins = set()
        for join in fanout.keys():
            joins.add(join[0])
            joins.add(join[1])
        joinkeys = {}
        for join in joins:
            table, col = join.split('.')
            joinkeys[TestDataset.ALIAS_TO_TABLE_NAME[table]] = col
        return joinkeys

    @staticmethod
    def LoadDataBase(database=None,
                    table=None, 
                    data_dir=datasets_path, 
                    try_load_parsed=True):
        """Loads a specified database's tables with a specified set of columns.

        Returns:
        A single CsvTable if 'database' and 'table' is specified, else a dict of CsvTables.
        """

        def TryLoad(table_name, filepath, use_cols, **kwargs):
            """Try load from previously parsed (table, columns)."""
            if use_cols:
                cols_str = '-'.join(use_cols)
                parsed_path = filepath[:-4] + '.{}.table'.format(cols_str)
            else:
                parsed_path = filepath[:-4] + '.table'
            if try_load_parsed:
                if os.path.exists(parsed_path):
                    arr = np.load(parsed_path, allow_pickle=True)
                    print('Loaded parsed Table from', parsed_path)
                    table = arr.item()
                    print(table)
                    return table
            print(TestDataset.tbls_cols_types[table_name])
            table = common.CsvTable(
                table_name,
                filepath,
                cols=use_cols,
                sep='|',
                keep_default_na=False,
                na_values=['NULL'],
                dtype=TestDataset.tbls_cols_types[table_name],
                **kwargs,
            )
            if try_load_parsed:
                np.save(open(parsed_path, 'wb'), table)
                print('Saved parsed Table to', parsed_path)
            return table

        def get_use_cols(filepath):
            return TestDataset.ContentColumns().get(filepath, None)
        
        if database and table:
            filepath = f'{database}/{table}.csv'
            table = TryLoad(
                table,
                data_dir + filepath,
                use_cols=get_use_cols(filepath),
                escapechar='\\',
                type_casts={},
            )
            return table

        tables = {}
        for filepath in TestDataset.TEST_DATASET_PRED_COLS:
            tables[filepath[0:-4]] = TryLoad(
                filepath[0:-4],
                data_dir + filepath,
                use_cols=get_use_cols(filepath),
                escapechar='\\',
                type_casts={},
            )

        return tables