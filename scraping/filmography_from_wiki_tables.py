from pathlib import Path
import polars as pl
from reusable import (
    join_multiindex, add_col, add_cols,
    extract_tables, first_argm, log, make_path_n_write,
    get_one_page, regex_year, clean_concat
)

URL_WIKI = 'https://en.wikipedia.org'


def _cast_cols(frame):
    return frame.select([pl.col('Year').cast(pl.UInt16), pl.exclude('Year').cast(pl.Utf8)])


def _last_lastname(strg):
    res = None
    if strg is not None:
        wrds = [w for w in strg.split() if w[0].isupper() and w[-1].islower()]
        if wrds:
            res = wrds[-1]
    return res


def _keep_former_if_not_null(col, suf='_temp'):
    return (
        pl.when(pl.col(col).is_null())
        .then(pl.col(f'{col}_right'))
        .otherwise(pl.col(col))
        .alias(f'{col}{suf}')
    )


def _first_max_n_uniques(frame, column, logger=None):
    cols = frame.columns
    assert cols
    if column not in cols:
        n_uniques = {col: frame.n_unique(subset=col) for col in cols}
        col = first_argm(n_uniques.items(), max)
        msg = (
            f'The table has no column `{column}` (of type lst[str]),'
            f' `{col}` will be used instead.')
        log(logger, msg)
    else:
        col = column
    return col


def _find_ytl_add_d(pd_df, logger=None):
    pl_df = pl.DataFrame(join_multiindex(pd_df))
    cols = pl_df.columns
    lst_str = pl_df.select(pl.col(pl.List(pl.Utf8)))

    # 'Year' and all_columns of type lst[str] but eventually 'Title'
    year = _first_max_n_uniques(
        lst_str.select(pl.all().arr[0].str.extract(regex_year[0]).cast(regex_year[1])),
        'Year',
        logger
    )
    frames = [lst_str.select(pl.all().arr[0]).select(pl.exclude('Title')).with_columns(
        pl.col(year).str.extract(regex_year[0]).cast(regex_year[1]).alias('Year')
    )]

    # 'Title' and 'Link'
    title = _first_max_n_uniques(lst_str.select(pl.all().arr[1]), 'Title', logger)
    assert 'Link' not in cols
    frames.append(
        lst_str.select([
            pl.col(title).arr[0].alias('Title'),
            (URL_WIKI + pl.col(title).arr[1]).alias('Link'),
        ])
    )

    # all_columns not of type lst[str] and eventually adds 'Director'
    frames.append(pl_df.select(sorted(set(cols) - set(lst_str.columns))))
    return _cast_cols(add_col(pl.concat(frames, how='horizontal'), 'Director'))


def _finalize(filmo, batch):
    keys = ['Year', 'Title', 'Director']

    if batch['director'] is not None:
        filmo = (
            filmo
            .select(pl.exclude('Director'))
            .with_columns(pl.lit(batch['director']).alias('Director'))
        )
    derived = 'Last Director Lastname'
    filmo = filmo.with_columns(
        pl.col('Director')
        .apply(_last_lastname, return_dtype=pl.Utf8)
        .alias(derived))

    arg = 'batch_name'
    if batch['sex'] is not None:
        pronouns = {'M': 'Himself', 'W': 'Herself'}
        filmo = filmo.with_columns(
            pl.col(keys[1:])
            .str
            .replace(pronouns[batch['sex']], batch[arg])
        )
    return _cast_cols(
        filmo
        .with_columns(pl.lit(batch[arg]).alias(arg))
        .select([pl.col(keys + [derived]), pl.exclude(keys + [derived])])  # reorder
        .unique(['Year', 'Title'], keep='first')
        # thus, unique on keys = ['Year', 'Title', 'batch_name']
        # cf. 1930 Anna Christie releases in German (Feyder) and English (Brown)
        # movies are later searched on ['Year', 'Title', 'batch_name']
        .sort(keys)
    )


def get_filmography(batch, save_path, logger=None):
    filmo = None
    log(logger, f'Getting filmography for batch: {batch}')

    if batch['tables'] is None:
        data = Path(batch['link']).read_text(encoding='utf-8').split('\n')
        filmo = pl.DataFrame(data, schema=['Link'])
        filmo = _cast_cols(add_cols(filmo, ['Year', 'Title', 'Director']))
        filmo = _add_missing_data(filmo, logger)

    elif batch['tables']:
        wiki_page = f"{URL_WIKI}/wiki/{batch['link']}"
        pd_dfs = get_one_page(extract_tables, wiki_page, logger, extract_links='body')
        if pd_dfs is not None:
            filmos = [
                _add_missing_data(_find_ytl_add_d(pd_df, logger), logger)
                for num, pd_df in enumerate(pd_dfs) if num in batch['tables']
            ]
            filmo = clean_concat(filmos, how='diagonal')

    if filmo is not None:
        filmo = _finalize(filmo, batch)
        make_path_n_write(filmo, save_path / f"{batch['batch_name']}.parquet")
    return filmo


def _get_first_infobox(url, logger=None):
    box = None
    log(logger, f'Looking for infobox at url {url}')
    boxes = get_one_page(extract_tables, url, logger, attrs={'class': 'infobox'})
    if boxes is not None:
        boxes = [box for box in boxes if box.shape[1] == 2]
        if boxes:
            box = pl.DataFrame(boxes[0])
        else:
            log(logger, f'There is no infobox at url {url}')
    return box


def _extract_field(box, field_name, sub_0, pat_1=None, logger=None):
    cols = box.columns
    fields = box.filter(pl.col(cols[0]).str.to_lowercase().str.contains(sub_0))
    if fields.is_empty():
        field = None
        log(logger, f'There is 0 {field_name}.', 'trace')
    else:
        if (length := fields.shape[0]) > 1:
            msg = f'There are {length} {field_name}s. The first one will be selected.'
            log(logger, msg, 'trace')
        fields = fields.select(pl.col(cols[1]))
        if pat_1 is not None:
            fields = fields.select(pl.col(cols[1]).str.extract(pat_1))
        field = fields.row(0)[0]
    return field


def _extract_infobox(url, logger=None):
    box = _get_first_infobox(url, logger)
    res = None
    if box is not None:
        title = box.columns[0]
        director = _extract_field(box, 'director', 'direc', None, logger)
        year = _extract_field(box, 'date', 'date', regex_year[0], logger)

        data = {'Link': url, 'Year': year, 'Title': title, 'Director': director}
        res = _cast_cols(pl.DataFrame(data))
    return res


def _add_missing_data(filmo, logger=None):
    keys = ['Year', 'Title', 'Director']
    missing_data = filmo.filter(pl.any(pl.col(keys).is_null()))
    if not missing_data.is_empty():
        log(logger, 'There is missing info, will now try to retrieve it.')
        missing_links = missing_data.filter(pl.col('Link').is_null())
        if not missing_links.is_empty():
            msg = (
                'There is no link for the following movies.'
                + str(missing_links.select(keys).to_dicts())
            )
            log(logger, msg)
        links = (
            missing_data
            .filter(pl.col('Link').is_not_null())
            .select('Link')
            .to_series()
            .to_list()
        )
        if links:
            found_data = [
                infobox for link in links
                if (infobox := _extract_infobox(link, logger)) is not None
            ]
            if found_data:
                added_data = (
                    missing_data
                    .join(pl.concat(found_data), on='Link', how='left')
                    .with_columns([_keep_former_if_not_null(key, '_temp') for key in keys])
                    .select(pl.exclude([f'{key}{suf}' for key in keys for suf in ['', '_right']]))
                    .rename({f'{key}_temp': key for key in keys})
                    .select(filmo.columns)  # reorder columns
                )
                not_missing_data = filmo.filter(pl.all(pl.col(keys).is_not_null()))
                filmo = pl.concat([not_missing_data, added_data])
        else:
            log(logger, 'There is no link at all for the missing info.')
    else:
        log(logger, 'There is no missing info.')
    return filmo


if __name__ == "__main__":
    from loguru import logger as logg

    xpl = {
        'link': 'Al_Pacino_on_stage_and_screen',
        'tables': [0],
        'batch_name': 'Al_Pacino',
        'sex': 'M',
        'director': None,
    }
    get_filmography(xpl, Path('./'), logg)
