import requests
from bs4 import BeautifulSoup


class LScript:
    def __init__(self, script_name):
        self.script = script_name
        self.langs = list()

    def add_lang(self, lang):
        self.langs.append(lang)


def get_langs_scripts(link):
    lang_scripts = []
    try:
        # get list of words
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        h2s = soup.find_all('h2')

        for idx, h2 in enumerate(h2s):
            h2span = h2.find_all('span', class_="mw-headline")
            # lang_scripts.append(h2span.a.text)
            lscript_ins = LScript(h2span.a.text)
            uls = h2.find_next_siblings('ul')
            for ul in uls:
                lis = ul.find_all('li')
                for li in lis:
                    lscript_ins.add_lang(li.a.text)
            lang_scripts.append(lscript_ins)
    except Exception as e:
        # errors can be due to timeouts/connections refused due to rate limiting. can be fixed via proxies/vpns/timeouts
        print("Error: ", e)
        return lang_scripts

    return lang_scripts


if __name__ == '__main__':
    wiki_link = 'https://en.wikipedia.org/wiki/List_of_languages_by_writing_system'
    l_s = get_langs_scripts(wiki_link)
    for i in l_s:
        print(len(i.langs))
        print(i.langs)