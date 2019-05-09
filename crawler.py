import requests
import json
import argparse
import threading
from queue import Queue

import pandas as pd
from bs4 import BeautifulSoup
import progressbar

BASE_URL = "http://www.dictionary.com/browse/%s?s=t"

def soupify(url):
    content = requests.get(url).content
    return BeautifulSoup(content, 'lxml')

def remove_style(entry):
    for style in entry.find_all('style'):
        style.extract()
    return entry

def remove_link(entry):
    for link in entry.find_all('a'):
        link.unwrap()
    return entry


def remove_example(entry):
    for example in entry.find_all('span', class_="luna-example italic"):
        example.extract()
    return entry


def extract_definition(entry):
    entry = remove_style(entry)
    entry = remove_link(entry)
    entry = remove_example(entry)
    definition = ' '.join([s for s in entry.stripped_strings])
    definition = definition[:-1] + "."
    return definition


def get_defs(word):

    url = BASE_URL % (word)
    soup = soupify(url)

    # find section that contains word definitions
    root = soup.find('section', class_="css-1748arg e1wu7xq20")
    if root is None:
        return None

    # find sub-sections
    pos_sections = root.find_all('section', class_="css-1sdcacc e10vl5dg0")

    defs = []
    for section in pos_sections:
        pos = section.find('span', class_="luna-pos")
        if pos:
            pos = pos.string.replace(',', '').split()[0]

        for entry in section.find_all('li', class_="css-2oywg7 e10vl5dg5"):
            definition = extract_definition(entry)
            defs.append((pos, definition))
            #print("  (%s)" % pos, definition)
    #print("")

    return defs


def main(args):

    with open(args.word_list) as f:
        word_list = json.load(f)

    pbar = progressbar.ProgressBar(max_value=len(word_list))
    word_queue = Queue()
    dictionary = {}

    plock = threading.Lock()
    global p
    p = 0

    def thread_func():
        global p
        while True:
            word = word_queue.get()
            defs = get_defs(word)
            dictionary[word] = defs
            word_queue.task_done()
            with plock:
                p = p + 1
                pbar.update(p)

    # Spawn threads
    for i in range(args.num_workers):
        t = threading.Thread(target=thread_func)
        t.daemon = True
        t.start()

    # Populate queue
    for word in word_list:
        word_queue.put(word)

    word_queue.join()

    with open(args.out, 'w') as f:
        json.dump(dictionary, f, ensure_ascii=False)

        
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--word-list', type=str,
                        default='data/glove6B_words.txt')
    parser.add_argument('--out', type=str,
                        default='dictionary.json')
    parser.add_argument('--num-workers', type=int, default=20)

    args = parser.parse_args()
    main(args)
