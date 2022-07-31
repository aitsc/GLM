import glob
from tqdm import tqdm


def books():  # formatted_one_article_per_line
    top_file = 0  # 提取前多少本书, 0表示全部
    books_path = './data/english_data/books1/epubtxt'  # 读取路径, 下面一个文件一本书
    output_filename = './data/english_data/books1' + (f'.{top_file}' if top_file else '') + '.txt'  # 保存路径
    print('output_filename:', output_filename)

    with open(output_filename, 'w', encoding='utf8') as ofile:
        paths = glob.glob(books_path + '/' + '*.txt', recursive=True)
        if top_file:
            paths = paths[:top_file]
        for i, filename in tqdm(enumerate(paths), 'books'):
            with open(filename, mode='r', encoding='utf-8-sig') as file:
                for line in file:
                    if line.strip() != '':
                        ofile.write(line.strip() + ' ')
            ofile.write("\n")
            # if i > 10:  # 后面合成一行测试
            #     ofile.write(" ")
            # else:
            #     ofile.write("\n")


def wiki():  # formatted_one_article_per_line
    top_n = 0  # 提取前多少文章, 0表示全部
    books_path = './data/english_data/wiki'  # 读取路径, 下面一个文件一批文章
    output_filename = books_path + (f'.{top_n}' if top_n else '') + '.txt'  # 保存路径
    print('output_filename:', output_filename)

    with open(output_filename, 'w', encoding='utf8') as ofile:
        paths = glob.glob(books_path + '/wiki_*', recursive=True)
        save_line_num = 0
        for filename in sorted(paths):
            print(filename)
            with open(filename, mode='r', encoding='utf-8-sig') as file:
                pbar = tqdm(desc='wiki')
                while True:
                    text = []
                    jump = True
                    for line in file:
                        line = line.strip()
                        if line[:9] == '<doc id="' and line[-2:] == '">':
                            jump = False
                            break
                    for line in file:
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        if line == '</doc>':
                            jump = False
                            break
                        text.append(line)
                    if len(text) > 1:
                        text = ' '.join(text) + '\n'
                        ofile.write(text)
                        pbar.update(1)
                        save_line_num += 1
                    if jump or top_n and save_line_num >= top_n:
                        break
            if top_n and save_line_num >= top_n:
                break


def 合并语料():
    paths = [  # formatted_one_article_per_line
        './data/english_data/wiki.txt',
        './data/english_data/books1.txt',
    ]
    ouput_path = './data/english_data/wikibook.txt'
    print('output_filename:', ouput_path)
    with open(ouput_path, 'w', encoding='utf8') as w:
        for p in paths:
            print(p)
            with open(p, 'r', encoding='utf8') as r:
                for line in tqdm(r, '合并语料'):
                    w.write(line.strip() + '\n')


if __name__ == '__main__':
    # books()
    wiki()
    # 合并语料()