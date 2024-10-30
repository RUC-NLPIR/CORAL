import os
import json
import re
from typing import List, Dict
from itertools import chain
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from pyspark import SparkContext, SparkConf,SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.types import (StringType, ArrayType, DoubleType,
                    StructField, StructType, IntegerType)
from pyspark.sql import functions as F
from bs4 import BeautifulSoup
import html

stops = set(stopwords.words('english')).union(set(",.{}[]-+:;!?/`~''\"()<>@#$%^&*|\\"))

class TreeNode:
    def __init__(self, name,level,passages=None):
        self.name = name
        self.level = level
        self.children = []  # 子节点列表
        self.parent = None  # 父节点
        self.next_sibling = None  # 指向下一个兄弟节点的引用
        self.passages = passages


keywords_file_path = ""

with open(keywords_file_path, 'r') as f:
    a_keywords_list = json.load(f)

## 1. clean and separate data
def process_wiki_raw(line):
    overalldict = []

    wiki = json.loads(line)
    url = wiki.get("url")
    #print("url: ", url)
    title = wiki.get("title")
    #print("title: ", title)
    text = wiki.get("text")
    if title in a_keywords_list:
        return []
    if text == "":
        #print("??????")
        return []
    #print("text: ", text)
    references = wiki.get("references")
    #print("references: ", references)
    
    text = clean_text(text)
    root = parse_html_to_tree(text, title)
    #print("root: ", root)
    if not root:
        return []
    path = find_longest_path(root)
    if len(path) <= 7:
        return []

    overalldict.clear()
    keywords_list = []
    path_list = []
    for sec in path:
        dict = {}
        keywords_list = []
        path_list.append(sec.name)
        temp_node = sec
        while temp_node:
            keywords_list = [temp_node.name] + keywords_list
            temp_node = temp_node.parent
        dict["keywords"] = keywords_list.copy()
        dict["path"] = path_list.copy()
        if sec.passages == None:
            continue
        match = re.search(r'(.*?)<h', sec.passages)
        if match:
            dict["passages"] = match.group(1)
        else:
            dict["passages"] = sec.passages
        dict["passages"] = clean_text_again(dict["passages"])

        #print(dict)
        overalldict.append(dict)

    #print(overalldict)
    
    ret = []
    for dict in overalldict:
        keywords = dict["keywords"]
        passages = dict["passages"]
        path = dict["path"]
        if passages == None:
            continue
        ref_ids = re.findall(r'\[[0-9]+\]', passages)
        cur_refs = []
        for ref in references:
            ref_id = ref["cite_id"]
            if ref_id not in ref_ids:
                continue
            for sub_ref in ref["sub_ref"]:
                processed_ref = process_ref(sub_ref)
                if processed_ref is None:
                    continue
                else:
                    ref_title, ref_content, ref_url = processed_ref
                    cur_refs.append([ref_id, ref_url, ref_title, ref_content])
        ret.append([url, keywords, passages, cur_refs, path])
    return ret

'''

    for i, section in enumerate(sections):
        keywords = [title]
        if i > 0:
            keywords.append(headings[i-1])
        
        ref_ids = re.findall(r'\[[0-9]+\]', section)
        cur_refs = []
        for ref in references:
            ref_id = ref["cite_id"]
            if ref_id not in ref_ids:
                continue
            for sub_ref in ref["sub_ref"]:
                processed_ref = process_ref(sub_ref)
                if processed_ref is None:
                    continue
                else:
                    ref_title, ref_content, ref_url = processed_ref
                    cur_refs.append([ref_id, ref_url, ref_title, ref_content])
        ret.append([url, keywords, section, cur_refs])

    return ret
'''



def clean_text(text):
    text = re.sub(r"(<a href=\"http(s)*\S+\">)|(</a>)", "", text)
    text = re.sub(r"&amp", "", text)
    text = re.sub(r"Coordinates :(.+?)[,\n.]", "", text) 
    #text = re.sub(r"Coordinates :(.+)\n", "", text) 
    text = re.sub("You can help Wikipedia by expanding it .", "", text)
    text = re.sub(r"&#(\d+);|<0x(.{1,3})>|\\xa0", " ", text)
    text = re.sub(r"(\.\S+)-(\S+)", " ", text)
    text = re.sub(r".longitude|.latitude", "", text)
    text = re.sub(r"{\S+}", " ", text)
    #text = re.sub(r"\(\S+\)", " ", text)
    text = re.sub(r"may refer to:|You can help Wikipedia by expanding it .", " ", text)
    text = re.sub(r"\[citation needed\]|\[edit\]|\[ edit\ ]", " ", text)
    text = re.sub(r"url\(.+\)", " ", text)
    text = re.sub(r" +", " ", text)
    
    return text

def clean_text_again(text):
    text = re.sub(r"(<a href=\"http(s)*\S+\">)|(</a>)", "", text)
    text = re.sub(r"&amp", "", text)
    text = re.sub(r"Coordinates :(.+?)[,\n.]", "", text) 
    #text = re.sub(r"Coordinates :(.+)\n", "", text) 
    text = re.sub("You can help Wikipedia by expanding it .", "", text)
    text = re.sub(r"&#(\d+);|<0x(.{1,3})>|\\xa0", " ", text)
    text = re.sub(r"(\.\S+)-(\S+)", " ", text)
    text = re.sub(r".longitude|.latitude", "", text)
    text = re.sub(r"{\S+}", " ", text)
    text = re.sub(r"\(\S+\)", " ", text)
    text = re.sub(r"may refer to:|You can help Wikipedia by expanding it .", " ", text)
    text = re.sub(r"\[citation needed\]|\[edit\]|\[ edit\ ]", " ", text)
    text = re.sub(r"url\(.+\)", " ", text)
    text = re.sub(r" +", " ", text)
    
    return text


def parse_html_to_tree(html_text, title):
    root = TreeNode(title,1)
    spans = [obj for obj in re.finditer("<h2>", html_text)]
    if len(spans) > 0:
        root_end_idx = spans[0].start()
        root_text = html_text[:root_end_idx]
        real_start_point = root_end_idx
        root.passages = root_text
        #print("-----------------")
    else:
        real_start_point = 0
        root.passages = ""
    
    #print(root.name, root.passages)
    #print("-----------------")
    stack = []  # 初始化一个空栈
    stack.append(root)
    #print(html_text)
    soup = BeautifulSoup(html_text, 'html.parser')
    
    
    tags = soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6'])
    #print(tags)

    real_html_text = html_text
    for i, tag in enumerate(tags):
        #print("--------------------")
        level = int(tag.name[1])  # 获取标题的级别，例如<h2>是2
        node = TreeNode(tag.text, level)
        #print(node.name)
        
        real_html_text = real_html_text[real_start_point:]
        # 获取当前标签的位置
        
        
        start_tag = tag
        start_text = str(start_tag)
        decoded_string = html.unescape(start_text)
        start_index = real_html_text.find(decoded_string) + len(decoded_string)
        
        # 获取下一个标签的位置
        end_tag = tags[i+1] if i+1 < len(tags) else None
        #print(end_tag)
        if end_tag:
            end_text = str(end_tag)
            decoded_string = html.unescape(end_text)
            end_index = real_html_text.find(decoded_string)
            #print(decoded_string)
            #print(start_index)
            #print(end_index)
            real_start_point = end_index
            text_between = real_html_text[start_index:end_index].strip()
        else:
            #print(start_index)
            #print(end_index)
            text_between = real_html_text[start_index:].strip()
        
            
        #print(start_index, end_index)
        #print(start_text)
        node.passages = text_between
        #print(node.passages)
        #print("-----------------")
        node.passages = text_between
        #print(node.passages)
        
        # 处理栈中的父子关系
        last_sibling = None
        while stack and stack[-1].level >= level:
            if not last_sibling and stack[-1].level == level:
                last_sibling = stack[-1]
            stack.pop()  # 如果当前标题的级别 <= 栈顶元素，弹出栈顶

        if stack:
            parent_node = stack[-1]  # 当前栈顶是父节点
            node.parent = parent_node  # 当前节点的父节点是栈顶元素
            parent_node.children.append(node)  # 将当前节点作为子节点加入父节点的children中
            if last_sibling:
                last_sibling.next_sibling = node
            

        stack.append(node)  # 将当前节点和级别入栈
        
       
    
    return stack[0] if stack else None  # 返回根节点


# 寻找最长路径（既有孩子节点又有兄弟节点的情况）
def find_longest_path(node):
    if not node:
        return []

    # 计算沿孩子节点的最长路径
    if node.children:
        longest_child_path = max([find_longest_path(child) for child in node.children], key=len, default=[])
    else:
        longest_child_path = []

    # 计算沿兄弟节点的最长路径
    if node.next_sibling:
        longest_sibling_path = find_longest_path(node.next_sibling)
    else:
        longest_sibling_path = []

    # 比较两种路径，选择较长的路径
    if len(longest_child_path) > len(longest_sibling_path):
        return [node] + longest_child_path
    else:
        return [node] + longest_sibling_path



# 解析HTML并构建树

def process_ref(ref, title=None):
    title = ref["title"]
    url = ref["url"]
    content = re.sub(r"\n|\s", " ", ref["text"])
    if len(content) < 128: 
        return None
    if not check_eng(content) & check_match(content, title):
        return None
    content = clean_text(content)
    return title, content, url


def check_eng(text):
    non_eng = re.findall(
        r'[^a-z_A-Z\d\.!@#\$%\\\^&\*\)\(\+=\{\}\[\]\/",\'<>~\·`\?:; |]', 
        text)
    if len(non_eng) / len(text) > 0.3:
        return False
    else:
        return True


def check_match(text, target):
    text = set(text.lower().split())
    target = set(target.lower().split())
    if target is None:
        return True
    elif len(text & target) > 0:
        return True
    else:
        return False


## 2. generate train data
"""
    input:
        text = str
    output:
        coarse_map = [ [sent, [id, ...]], ...]
"""
def coarse_ref_map(text):
    sentences = custom_sent_tokenize(text.lower())
    # print("coarse_ref_map sentences: ", sentences)
    rtn = []
    if len(sentences) == 1:
        return [[sentences[0], ["[0]"]]]
    for i in range(len(sentences)-1):
        # 如果下一个句子是以引用标识符开头。
        if re.match("\[[0-9]+\]", sentences[i+1]):
            # 如果当前句子不以引用标识符开头。
            if not re.match("\[[0-9]+\]", sentences[i]):
                # 将当前句子和下一个句子作为一个元素添加到结果列表中。
                rtn.append([sentences[i], [sentences[i+1]]])
            else:
                try:
                    rtn[-1][1].append(sentences[i+1])
                except:
                    continue
        elif not re.match("\[[0-9]+\]", sentences[i]):
           rtn.append([sentences[i], ["[0]"]])
        if i == len(sentences)-2 and not re.match("\[[0-9]+\]", sentences[i+1]):
            rtn.append([sentences[i+1], ["[0]"]])
    #print(rtn)
    return rtn


"""
    input:
        text = str
    output:
        coarse_map = [sent, ...]
"""
def custom_sent_tokenize(text):
    ori_sentences = sent_tokenize(text.replace("\n", ""))
    ori_sentences = list(
            chain(*[re.split("(\[[0-9]+\])", sent)    
             for sent in ori_sentences
            ]))
    ori_sentences = [sent for sent in ori_sentences if len(sent.strip()) != 0]
    # print(ori_sentences)
    sentences = []

    current_sent = ""
    for sent in ori_sentences:
        # print(sent)
        if re.match("\[[0-9]+\]", sent):

            if current_sent != "":
                sentences.append(current_sent)
                current_sent = "" 
            sentences.append(sent)

        elif sent.endswith(("e.g.","i.e.")) or len(sent) < 8:
            current_sent += f" {sent}"

        elif current_sent.endswith((".", "!", "?", ".\"", "?\"", "!\"")):
            current_sent += f" {sent}"
            sentences.append(current_sent)
            current_sent = ""

        else:
            current_sent += f" {sent}"
    
    if current_sent:
        sentences.append(current_sent)
    return sentences


"""input:
        coarse_map = [ [sent, [id, ...]], ...]
        refs = [ [id,url,title,text],...]
    output:
        coarse_map = [ [sent, [id, ...]], ...]
        rtn_passage = {id:[str, title], ...}
"""
def refine_ref_map(coarse_map, refs):
    
    ref_ids = set(list(chain(*[p[1] for p in coarse_map])))
    if ref_ids == set(["[0]"]) or len(refs) == 0:
        return coarse_map, {}
    ref_passage = defaultdict(list)
    for ref in refs:
        ref_id = ref[0]
        ref_title = ref[2]
        if ref_id not in ref_ids or len(ref[-1].strip()) == 0:
            continue
        ref_passage[ref_id].extend(split_text(f"{ref[3]}", 256))
        ref_passage[ref_id].append(ref_title)  # title line: 230+139+260+293
    
    current_ref_ids = []
    rtn_passage = {}
    trace_back_count = 0

    for i in range(len(coarse_map)-1, -1, -1): 
        if coarse_map[i][1] == ["[0]"] and not current_ref_ids:
            continue
        elif coarse_map[i][1] != ["[0]"]:
             current_ref_ids = coarse_map[i][1]
             trace_back_count = 0
        else:
            if trace_back_count > 3:
                continue
            check_ref = check_cited(coarse_map[i][0], ref_passage, current_ref_ids)
            if check_ref:
                coarse_map[i][1] = check_ref
                trace_back_count += 1
            else:
                trace_back_count += 3
    rtn_passage = rank_ref_passage(ref_passage, coarse_map)
    
    return coarse_map, rtn_passage


"""
    input:
        text = str
        maxlem = int
    output:
        sub_texts = [[w1, w2 ...], [w1, w2 ...] ...]
"""
def split_text(text, maxlen, greedy=False):
    sentences = sent_tokenize(text.lower())
    sentences = [word_tokenize(sent) for sent in sentences]
    sent_lens = [len(s) for s in sentences]
    if sum(sent_lens) <= maxlen:
        return [list(chain(*sentences))]
    n_sentences = len(sentences)
    alls = []  
    for i in range(n_sentences):
        length = 0
        sub = []
        for j in range(i, n_sentences):
            if length + sent_lens[j] <= maxlen or not sub:
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        alls.append(sub)
        if j == n_sentences - 1:
            if sub[-1] != j:
                alls.append(sub[1:] + [j])
            break

    if len(alls) == 1:
        return [list(chain(*sentences))]
    if greedy:  
        sub_texts = [list(chain(*[sentences[i] for i in sub])) for sub in alls]
        return sub_texts
    else:  # 用动态规划求解满足要求的最优子片段集
        DG = {}
        N = len(alls)
        for k in range(N):
            tmplist = list(range(k + 1, min(alls[k][-1] + 1, N)))
            if not tmplist:
                tmplist.append(k + 1)
            DG[k] = tmplist
        routes = {}
        routes[N] = (0, -1)
        for i in range(N - 1, -1, -1):
            templist = []
            for j in DG[i]:
                cross = set(alls[i]) & (set(alls[j]) if j < len(alls) else set())
                w_ij = sum([sent_lens[k] for k in cross]) ** 2  # 第i个节点与第j个节点交叉度
                w_j = routes[j][0]  # 第j个子问题的值
                w_i_ = w_ij + w_j
                templist.append((w_i_, j))
            routes[i] = min(templist)
        sub_texts = [list(chain(*[sentences[i] for i in alls[0]]))]
        k = 0
        while True:
            k = routes[k][1]
            sub_texts.append(list(chain(*[sentences[i] for i in alls[k]])))
            if k == N - 1:
                break
    return sub_texts


"""
    input:
        src = str
        refs = {id: [[w1, w2 ...], [w1, w2 ...], ..., title], ...}
        ref_ids = [id,...]
    output:
        rtn_ids [id,...]
"""
def check_cited(src, refs, ref_ids):
    rtn_ids = []
    src_toks = set(word_tokenize(src)) - stops
    if len(src_toks) == 0:
        return rtn_ids
    for ref_id in ref_ids:
        ref_toks = refs[ref_id]
        ref_toks = set(list(chain(*ref_toks))) - stops
        if len(ref_toks & src_toks)/ len(src_toks) > 0.3:
            rtn_ids.append(ref_id)
    return rtn_ids


"""
    input:
        ref_passage = {id: [[w1, w2 ...], [w1, w2 ...] ...title], ...}
        refs_map = [ [sent, [id, ...]], ...]
    output:
        rtn = {id:[str, title], ...}
"""
def rank_ref_passage(ref_passage, ref_map):
    rtn = {}
    src_toks_by_ref = defaultdict(list)
    for sent in ref_map:
        for ref_id in sent[1]:
            src_toks_by_ref[ref_id].extend(word_tokenize(sent[0]))
    for ref_id in ref_passage:
        tok_count = []
        for p in ref_passage[ref_id][:-1]:
            n = compute_match(p, src_toks_by_ref[ref_id])
            tok_count.append(n)
        if len(tok_count) == 0:
            continue
        passage_idx = tok_count.index(max(tok_count))
        rtn[ref_id] = [" ".join(ref_passage[ref_id][:-1][passage_idx]), ref_passage[ref_id][-1]]

    return rtn

"""
    input:
        src = [w1, w2 ...]
        trg = [w1, w2 ...]
    output:
        rtn = int
"""
def compute_match(src, trg) -> int:
    src_toks = set(src) - stops
    trg_toks = set(trg) - stops
    return len(src_toks & trg_toks)


## 3. after process

"""
    input:
        coarse_map = [ [sent, [id, ...]], ...]
        rtn_passage = {id:[str,title], ...}
    output:
        text = [[sent, id, ...], ...]
        refs = [[id, title, str], ... ]
"""
def after_process(coarse_map, rtn_passage):
    text, refs = [], []
    
    for sent in coarse_map:
        text.append([sent[0]])
        text[-1].extend(sent[1])
    for k,v in rtn_passage.items():
        refs.append([k,v[1],v[0]])
    refs.sort(key=lambda x:[x[0]])
    return text, refs


# main
def main():

    dataset_path = "" 
    cur_save_path = ""  # 运行前一定不能有这个dir不然会报错

    with open(dataset_path, 'r') as rfile,open(cur_save_path, "w") as wfile:
        for line in rfile:
            # read raw file
            # process raw file
            data = process_wiki_raw(line)       
            # save
            # x = [url, title, text, refs]
            for x in data:
                url, keywords, passages, cur_refs ,path = x
                passages = coarse_ref_map(passages)
                passages, cur_refs = refine_ref_map(passages, cur_refs)
                text,refs = after_process(passages, cur_refs)


                dict = {}
                dict["url"] = url
                dict["title"] = keywords 
                dict["text"] = text
                dict["references"] = refs
                dict["path"] = path 
                json_string = json.dumps(dict)
                wfile.write(json_string)
                wfile.write('\n')


               
                
  
    print("finish!")

'''
    dict = {}
    with open(dataset_path, 'r') as rfile,open(cur_save_path, "w") as wfile:
        for line in rfile:
            # read raw file
            # process raw file
            data = process_wiki_raw(line)       
            # save
            # x = [url, title, text, refs]
            for x in data:
                x[2] = coarse_ref_map(x[2])
                x[2], x[3] = refine_ref_map(x[2], x[3])
                x[2], x[3] = after_process(x[2], x[3])
                dict["keywords"] = x[1]
                dict["passages"] = x[2]
                dict["refs"] = x[3]
                dict["url"] = x[0]
                #print(len(dict["passages"][0]))

                #print(dict["keywords"])       
                if len(dict["passages"]) == 0:
                    continue
                
                elif len(dict["passages"]) == 1 and len(dict["passages"][0][0]) < 32:
                    continue
            
                json_string = json.dumps(dict)
                wfile.write(json_string)
                wfile.write('\n')
                dict = {}
    print("finish!")
'''

if __name__ == '__main__':
    main()
