from re import sub
from requests import get
from bs4 import BeautifulSoup
import pandas as pd

def clean_string(string):
    return (
        sub('\[[A-Z]+\]', '', sub('\[\d+\]', '', string))
        .replace('\n', '').replace('\xa0', ''))

def get_text_line(lst):
    return [clean_string(ele.text) for ele in lst]

def parse_table(url, n_table):
    html = BeautifulSoup(get(url).text, features='html.parser')
    table = table_to_frame(html.find_all('table')[n_table])
    return table

def table_to_frame(table):
    lines = table.find_all('tr')
    columns = get_text_line(lines[0].find_all('th'))
    n_cols = len(columns)
    lines = [get_text_line(line.find_all('td')) for line in lines[1:]]
    lines = [
        line if len(line) == n_cols else [None] + line
        for line in lines]
    assert all(len(line) == n_cols for line in lines)

    frame = pd.DataFrame(columns=columns, data=lines)
    frame.iloc[:, 0] = frame.iloc[:, 0].ffill()
    return frame

URL = 'https://en.wikipedia.org/wiki/List_of_circulating_currencies'
currencies = parse_table(URL, 0)
print(dict(currencies.groupby('ISO code')['State or territory'].apply(set)))
