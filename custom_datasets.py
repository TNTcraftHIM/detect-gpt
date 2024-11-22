import random
import datasets
import pandas as pd
import tqdm

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed', 'raid', 'csv']


def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub

def process_text_truthfulqa_adv(text):
    if not type(text) == str:
        return ''

    if "I am sorry" in text or "I'm sorry" in text:
        try:
            first_period = text.index('.')
        except ValueError:
            try:
                first_period = text.index(',')
            except ValueError:
                first_period = -2
        start_idx = first_period + 2
        text = text[start_idx:]
    # if "as an AI language model" in text or "As an AI language model" in text:
    if "as an AI language model" in text or "As an AI language model" in text or "I'm an AI language model" in text or "As a language model" in text:
        try:
            first_period = text.index('.')
        except ValueError:
            first_period = text.index(',')
        start_idx = first_period + 2
        text = text[start_idx:]
    return text

## Use global variable to cache the data to avoid loading the dataset multiple times
RAID_Cache = {}
RAID_CSV_Cache = None
def load_raid(cache_dir, source, attack='none', train_ratio=0.8):
    global RAID_Cache, RAID_CSV_Cache

    # 如果缓存中已经有对应 LLM_name 的数据，直接返回缓存的结果
    if source != 'machine' and source in RAID_Cache:
        # 随机打乱数据
        random.shuffle(RAID_Cache[source]['train']['text'])
        random.shuffle(RAID_Cache[source]['test']['text'])
        return RAID_Cache[source].copy()

    if RAID_CSV_Cache is None:
        # 读取 CSV 文件
        f = pd.read_csv("data/RAID_train.csv")
        RAID_CSV_Cache = f
    else:
        f = RAID_CSV_Cache

    # 筛选指定模型的数据
    if source == 'machine':
        ## Select a model from the list of models except for human
        selected_model = random.choice([_ for _ in f['model'].unique() if _ != 'human'])
        selected_data = f[f['model'] == selected_model]
    else:
        selected_data = f[f['model'] == source]
    if attack != 'any' and source != 'human':
        selected_data = selected_data[selected_data['attack'] == attack]

    # 提取问题、答案和类别
    q = selected_data['title'].tolist()
    a = selected_data['generation'].fillna("").tolist()
    a = [process_text_truthfulqa_adv(_) for _ in a]  # 预处理
    c = selected_data['domain'].tolist()  # 类别信息

    # 生成结果列表
    res = []
    for i in range(len(q)):
        if len(a[i].split()) > 5 and len(a[i].split()) < 150:  # 筛选较长的答案
            res.append([q[i], a[i], c[i]])
            # 确保答案以句号结尾
            if res[-1][1][-1] != '.':
                res[-1][1] += '.'

    # 创建新的数据结构
    data_new = {
        'train': {
            'text': [],
            'label': [],
            'category': [],
        },
        'test': {
            'text': [],
            'label': [],
            'category': [],
        }
    }

    # 随机打乱数据
    random.shuffle(res)
    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        # 根据比例划分训练集和测试集
        data_partition = 'train' if i < total_num * train_ratio else 'test'

        # 添加问题、回答和类别
        data_new[data_partition]['text'].append(process_spaces(res[i][1]))
        data_new[data_partition]['label'].append(0 if source == 'human' else 1)  # 0 为人类答案，1 为机器答案
        data_new[data_partition]['category'].append(res[i][2])

    # 将结果缓存到 RAID_Cache 中
    RAID_Cache[source] = data_new.copy()

    return data_new['train']['text']

def load_csv(cache_dir):
    # 从 csv 文件加载数据集
    data = pd.read_csv('data/test_data.csv', encoding='utf-8')
    return data

def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')