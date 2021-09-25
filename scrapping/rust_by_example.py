from os import mkdir, chdir
from subprocess import Popen
from collections import defaultdict
from requests import get
from bs4 import BeautifulSoup

BASE_URL = 'https://doc.rust-lang.org/rust-by-example/'
text = BeautifulSoup(get(BASE_URL + 'index.html').text, features='html.parser')
links = [link.get('href') for link in text.find_all('a', href=True)][1:192]

# put the links in different groups
old = None
count = 0
groups = defaultdict(list)
for link in links:
    if (beginning := link[:-5].split('/')[0]) != old:
        old = beginning
        count += 1
    groups[str(count).zfill(2) + '_' + old].append(link)

# load the rust snippets in each html page and compile them
for group, links in groups.items():
    mkdir(group)
    chdir(group)
    for link in links:
        content = BeautifulSoup(get(BASE_URL + link).text, features='html.parser')
        snippets = content.find_all('code', {'class': "language-rust editable"})
        for num, snippet in enumerate(snippets):
            full_name = '_'.join(link.split('/'))[:-5] + '_' + str(num) + '.rs'
            file = open(full_name, 'w')
            file.write(snippet.text)
            file.close()
            proc = Popen(['rustc', full_name])
            proc.communicate()
            if (retcode := proc.returncode) != 0:
                print(f'{full_name}: error {retcode} while compiling.')
    chdir('../')
