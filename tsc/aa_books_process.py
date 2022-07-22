import glob
from tqdm import tqdm

top_file = 0
books_path = './data/english_data/books1/epubtxt'
output_filename = './data/english_data/books1' + (f'.{top_file}' if top_file else '') + '.txt'

with open(output_filename, 'w', encoding='utf8') as ofile:
    paths = glob.glob(books_path + '/' + '*.txt', recursive=True)
    if top_file:
        paths = paths[:top_file]
    for filename in tqdm(paths):
        with open(filename, mode='r', encoding='utf-8-sig') as file:
            for line in file:
                if line.strip() != '':
                    ofile.write(line.strip() + ' ')
        ofile.write("\n\n")