from re import sub  #, search
from requests import get
from bs4 import BeautifulSoup
from numpy import isnan
import pandas as pd

def clean_string(string):
    return (
        sub(r'\[[A-Z]+\]', '', sub(r'\[\d+\]', '', string))
        .replace('\n', '').replace('\xa0', ''))

def clean_text_line(lst):
    return [clean_string(ele.text) for ele in lst]   # if lst else [None]

def get_colspan(col):
    return 1 if (colspan_ := col.get('colspan')) is None else int(colspan_)

def parse_index(table):
    rows = table.find_all('tr')
    has_multiindex = any(get_colspan(col) != 1 for col in rows[0].find_all('th'))
    if has_multiindex:  # TODO: here we assumed that the multiindex spans on 2 columns
        columns = []
        for col in rows[0].find_all('th'):
            colspan = get_colspan(col)
            columns += [(clean_string(col.text), colspan > 1)] * colspan
        spanned_columns = [(*desc, num) for num, desc in enumerate(columns) if desc[1]]
        for sub_col, spanned_col in zip(rows[1].find_all('th'), spanned_columns):
            columns[spanned_col[2]] = spanned_col[0] + ' ' + clean_string(sub_col.text)
        columns = [col[0] if isinstance(col, tuple) else col for col in columns]
        lines = rows[2:]
    else:
        columns = clean_text_line(rows[0].find_all('th'))
        lines = rows[1:]
    return columns, lines

def parse_line(line, n_cols):
    parsed_data = None
    seps = line.find_all(['td', 'th'])
    if len(seps) > 1:
        parsed_data = clean_text_line(seps)
        pos = 1 if len(seps) == n_cols else 0
        link = link_.get('href') if (link_ := seps[pos].find('a')) is not None else None
        parsed_data.insert(pos + 1, link)
        if pos == 0:
            parsed_data.insert(0, None)
    return parsed_data

def table_to_frame(table):
    columns, lines = parse_index(table)
    n_columns = len(columns)

    data = [
        parsed_line for line in lines
        if (parsed_line := parse_line(line, n_columns)) is not None]
    for num, line in enumerate(data):
        if len(line) != n_columns + 1:
            print(f'Removing the line {num}: {line}.')
    data = [line for line in data if len(line) == n_columns + 1]
#     assert all(len(line) == n_columns + 1 for line in data)

    columns.insert(2, 'Link')
    frame = pd.DataFrame(columns=columns, data=data)
    frame.iloc[:, 0] = frame.iloc[:, 0].ffill()  # trick: index is ordered
    base_url = 'https://en.wikipedia.org'
    to_edit = '/w/index.php'
    if 'Link' in frame:
        frame['Link'] = frame['Link'].apply(
            lambda link: base_url + link
            if link and not link.startswith(to_edit) else None)
    return frame

def parse_tables(url, n_tables):
    if isinstance(n_tables, int):
        n_tables = [n_tables]
    html = BeautifulSoup(get(url).text, features='html.parser')
    tables = html.find_all('table')
    frames = [table_to_frame(tables[n_table]) for n_table in n_tables]
    return pd.concat(frames).sort_values(by=frames[0].columns[0]).reset_index(drop=True)

def select_line(frame, keyword):
    string = None
    if frame is not None:
        content = frame[frame.iloc[:, 0].str.lower().str.contains(keyword)]
        if not content.empty:
            string = clean_string(content.iloc[0, 1])
    return string

def select_fields(frame):
    series = pd.Series({'Title': None, 'Director': None, 'Year': None})
    if frame is not None and not frame.empty:
        series = pd.Series({
            'Title': frame.columns[1],
            'Director': select_line(frame, 'direct'),
            'Year': select_line(frame, 'date')})
    return series

def read_infobox(wiki_url):
    infos = None
    if bool(wiki_url) and not isnan(wiki_url) and (wiki_url := str(wiki_url)).startswith('http'):
        print(f'Checking the page: {wiki_url}')
        html = BeautifulSoup(get(wiki_url).text, features='html.parser')
        tables = html.find_all('table')
        if not tables:
            print(f'The page {wiki_url} contains no table.')
        elif (infoboxes := [
            table for table in tables
            if table.has_attr('class') and 'infobox' in table['class']]):
            infos = table_to_frame(infoboxes[0])
        else:
            print(f'The page {wiki_url} contains no infobox.')
    return infos

def read_infoboxes(wiki_urls, fields):
    infos = [select_fields(read_infobox(wiki_url)) for wiki_url in wiki_urls]
    return pd.concat(infos, axis=1).loc[fields, :].T.reset_index(drop=True)

def reorder_list(sublst, lst):
    assert set(sublst) <= set(lst)  # quid repetitions?
    return sublst + [col for col in lst if col not in sublst]

def lastname_last_director(strg):
    res = None
    if strg is not None and (wrds := [w for w in strg.split() if w[0].isupper()]):
        res = wrds[-1]
    return res

def carefully_replace_column_name(cols, old_col, new_col):
    if old_col in cols and new_col != old_col:
        if new_col in cols:
            # eventually several columns new_col.upper()
            cols = cols.str.replace(new_col, new_col.upper())
        cols = cols.str.replace(old_col, new_col)
    return cols
