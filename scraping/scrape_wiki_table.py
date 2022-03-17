from re import sub
from requests import get
from bs4 import BeautifulSoup
import pandas as pd

def clean_string(string):
    return (
        sub('\[[A-Z]+\]', '', sub('\[\d+\]', '', string))
        .replace('\n', '').replace('\xa0', ''))

def get_text_line(lst):
    return [clean_string(ele.text) for ele in lst]   # if lst else [None]

def parse_index(table, index_width):
    rows = table.find_all('tr')
    # deals with the names of the columns: this index can be spanned over two columns
    # rows[0], rows[1]
    cols = []
    for col in rows[0].find_all('th'):
        colspan = col.get('colspan')
        colspan = 1 if colspan is None else int(colspan)
        cols += [(clean_string(col.text), colspan > 1)] * colspan

    assert index_width <= 2, 'index_width > 2 is not implemented'
    if index_width == 2:
        spanned_cols = [(*desc, num) for num, desc in enumerate(cols) if desc[1]]
        for sub_col, spanned_col in zip(rows[1].find_all('th'), spanned_cols):
            cols[spanned_col[2]] = spanned_col[0] + ' ' + clean_string(sub_col.text)

    cols = [col[0] if isinstance(col, tuple) else col for col in cols]
    return rows[index_width:], [col.replace('\n', '') for col in cols]

def parse_line(line, shift, n_cols):
    seps = line.find_all('td')
    parsed_data = None
    if len(seps) > 1:
        parsed_data = get_text_line(seps)
        link = link_.get('href') if (link_ := seps[shift].find('a')) is not None else ''
        parsed_data.insert(1 + shift, link)
        if shift == 0 and (th_ := line.find('th')) is not None:
            parsed_data = get_text_line([th_]) + parsed_data
        if len(parsed_data) < n_cols:
            parsed_data.insert(0, None)
    return parsed_data

def table_to_frame(table, index_width):
    lines, cols = parse_index(table, index_width)
    cols.insert(2, 'Link')
    n_cols = len(cols)

    for line in lines:
        sep = line.find_all('td')
        if len(sep) > 1:
            break
    sep_is_only_td = len(sep) + 1 == n_cols
    shift = 1 if sep_is_only_td else 0
    data = [
        parsed for line in lines if
        (parsed := parse_line(line, shift, n_cols)) is not None]

    assert all(len(line) == n_cols for line in data)
#     for num, line in enumerate(data):
#         if len(line) != n_cols:
#             print(f'Removing the line: {line}.')
#     data = [line for line in data if len(line) == n_cols]

    frame = pd.DataFrame(columns=cols, data=data)
    frame.iloc[:, 0] = frame.iloc[:, 0].ffill()
    frame['Link'] = 'https://en.wikipedia.org' + frame['Link']

    return frame

def parse_tables(url, n_tables=0, index_widths=1):
    if isinstance(n_tables, int):
        n_tables = [n_tables]
        index_widths = [index_widths]
    assert len(n_tables) == len(index_widths)

    html = BeautifulSoup(get(url).text, features='html.parser')
    tables = html.find_all('table')
    frames = [
        table_to_frame(tables[n_table], width)
        for n_table, width in zip(n_tables, index_widths)]  # strict=True
    return pd.concat(frames).sort_values(by=frames[0].columns[0])

currencies = parse_tables('https://en.wikipedia.org/wiki/List_of_circulating_currencies')
dict(currencies.groupby('ISO code')['State or territory'].apply(set))
