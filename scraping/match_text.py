from itertools import product
import numpy as np
import pandas as pd
# from polyleven import levenshtein as levenshtein_dist

def levenshtein_dist(str_1, str_2):
    n_rows = len(str_1) + 1
    n_cols = len(str_2) + 1
    dists = np.zeros((n_rows, n_cols)).astype(int)
    dists[:, 0] = range(n_rows)
    dists[0, :] = range(n_cols)
    for n_row, n_col in product(range(1, n_rows), range(1, n_cols)):
        sub_cost = 0 if str_1[n_row - 1] == str_2[n_col - 1] else 1
        dists[n_row][n_col] = min(
            dists[n_row][n_col - 1] + 1,  # insertion
            dists[n_row - 1][n_col] + 1,  # deletion
            dists[n_row - 1][n_col - 1] + sub_cost  # substitution
            )
    return dists[n_rows - 1][n_cols - 1]

def get_line_ind(series, string):
    return series.apply(lambda x: levenshtein_dist(
        str(x).lower(), str(string).lower())).argmin()

def get_line(frame, col, string):
    return frame.iloc[get_line_ind(frame[col], string), :]

def reorder_list(sublst, lst):
    assert set(sublst) <= set(lst)  # quid repetitions?
    return sublst + [col for col in lst if col not in sublst]

def match_frames(to_match, records, through, ordered_cols_sublst=None):
    assert records.index.is_unique, 'index of records must be uniquely valued'
    assert list(to_match.index) == list(range(len(to_match))), (
        'index of to_match must be the default index')
    assert not (set(to_match) & set(records)), (
        'the columns of the dataframes must be pairwise different')

    to_records = lambda string: get_line_ind(records[through[1]], string)
    indexes_in_records = list(to_match[through[0]].apply(to_records))
    matched_records = records.iloc[indexes_in_records].reset_index(drop=True)
    matched = pd.concat([matched_records, to_match], axis=1)
    matched['not_matched'] = False

    not_matched_indexes = sorted(set(records.index) - set(indexes_in_records))
    merged = pd.concat([matched, records.iloc[not_matched_indexes]])
    merged['not_matched'] = merged['not_matched'].fillna(True)

    if ordered_cols_sublst is not None:
        ordered_cols = reorder_list(ordered_cols_sublst, merged.columns)
        merged = merged.sort_values(by=ordered_cols_sublst).loc[:, ordered_cols]
    return merged.reset_index(drop=True)
