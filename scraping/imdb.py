import polars as pl
from bs4 import BeautifulSoup
from reusable import regex_year, log, get_one_page, crawl, clean_concat


def gen_url(start, val, field):  # fields = groups, genres, ...
#     params = {field: val, 'start': start, 'ref_': 'adv_nxt'} could be passed to requests.get
    return f"https://www.imdb.com/search/title/?{field}={val}&start={start}&ref_=adv_nxt"


def extract_data(res, logger):
    # regexes: https://www.freecodecamp.org/news/web-scraping-sci-fi-movies-from-imdb-with-python/
    html = BeautifulSoup(res.text, 'html.parser')
    log(logger, f'Extracting content of {res.url}')
    containers = html.find_all('div', class_='lister-item mode-advanced')
    fields = [
        ['h3', 'span', {'class_': 'lister-item-year text-muted unbold'}, 'year'],
        ['p', 'span', {'class_': 'certificate'}, 'rating'],
        ['p', 'span', {'class_': 'genre'}, 'genre'],
        ['p', 'span', {'class_': 'runtime'}, 'runtime'],
    ]
    fields_names = [f[3] for f in fields] + ['title', 'imdb_rating', 'm_score', 'vote']
    dic = {field: [] for field in fields_names}

    for container in containers:
        if container.find('div', class_='ratings-metascore') is not None:
            dic['title'].append(container.h3.a.text)

            for field in fields:
                ext = getattr(container, field[0]).find(field[1], **field[2])
                dic[field[3]].append(ext.text if ext is not None else None)

            ext = container.strong
            dic['imdb_rating'].append(ext.text if ext is not None else None)

            ext = container.find('span', class_='metascore')
            dic['m_score'].append(ext.text if ext is not None else None)

            ext = container.find('span', attrs={'name':'nv'})
            dic['vote'].append(ext.get('data-value', None) if isinstance(ext, dict) else None)

    regexes = [
        ['year', *regex_year],
        ['runtime', r'(\d+)', pl.UInt16],
        ['imdb_rating', r'(\d+.\d+)', pl.Float32],
        ['m_score', r'(\d+)', pl.UInt8],
        ['vote', r'(\d+)', pl.UInt32],
    ]
    return (
        pl.DataFrame(dic)
        .with_columns(pl.all().cast(pl.Utf8))
        .with_columns([
            pl.col(col).str.extract(regex).cast(dtype)
            for col, regex, dtype in regexes
        ])
    )


if __name__ == "__main__":
    from itertools import count
    from pathlib import Path
    from loguru import logger as logg

    groups = [
        'top_100',
        'top_250',
        'top_1000',
        'bottom_100',
        'bottom_250',
        'bottom_1000'
        'oscar_winner',
        'emmy_winner',
        'golden_globe_winner',
        'oscar_nominee',
        'emmy_nominee',
        'golden_globe_nominee',
        'best_picture_winner',
        'best_director_winner',
        'oscar_best_picture_nominees',
        'oscar_best_director_nominees',
        'national_film_preservation_board_winner',
        'razzie_winner',
        'razzie_nominee',
    ]

    data_groups = []
    for group in groups:
        iterator = count(1, 50)
        # https://www.imdb.com/search/title: default display option 50 per page
        save_path = Path(f'./imdb/{group}.parquet')
        def get_datum(name):
            return get_one_page(extract_data, gen_url(name, group, 'groups'), logg)
        data_groups.append(crawl(get_datum, iterator, save_path, 1, logg))
    clean_concat(data_groups).write_parquet(Path('./imdb/groups.parquet'))
