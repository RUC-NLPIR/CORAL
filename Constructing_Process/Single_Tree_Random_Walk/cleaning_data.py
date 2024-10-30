import os
import json
import re
from typing import List, Dict
from itertools import chain
from collections import defaultdict
import random
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from pyspark import SparkContext, SparkConf,SparkFiles
from pyspark.sql import SparkSession
from pyspark.sql.types import (StringType, ArrayType, DoubleType,
                    StructField, StructType, IntegerType)
from pyspark.sql import functions as F
from anytree import Node, RenderTree,NodeMixin,PreOrderIter
from collections import deque

stops = set(stopwords.words('english')).union(set(",.{}[]-+:;!?/`~''\"()<>@#$%^&*|\\"))

class MyBaseClass(object):  # Just an example of a base class
    foo = 4


class MyClass(MyBaseClass, NodeMixin):  # Add Node feature
    def __init__(self, title, passages, level, parent=None,children=None):
        super(MyClass, self).__init__()
        self.title = title
        self.passages = passages
        self.level = level
        self.parent = parent
        if children:  # set children only if given
            self.children = children
        








## 1. clean and separate data
def process_wiki_raw(line):
    wiki = json.loads(line)
    url = wiki.get("url")
    #print("url: ", url)
    title = wiki.get("title")
    #print("title: ", title)
    text = wiki.get("text")
    if text == "":
        #print("??????")
        return []
    #print("text: ", text)
    references = wiki.get("references")
    #print("references: ", references)
    
    text = clean_text(text)

    root = MyClass(title, text,1)
    build_tree(root)

    '''
    for child in root.children:
        #print(child.title)
        if child.title == "overview":
            print(hasattr(child, 'children'))
            print(not child.children)
            print("---------")
    '''           
    
    #bfs(root)
    


  
    
    # 获取所有节点
    all_nodes = get_all_nodes(root)
    

    # 选择节点
    overalldict = process_nodes(all_nodes)
    #print(len(overalldict))

    if len(overalldict) < 10:
        return []
    
    
    ret = []
    for dict in overalldict:
        #print(dict["keywords"])
        keywords = dict["keywords"]
        passages = dict["passages"]
        path = dict["path"]
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
        ret.append([url, keywords, passages, cur_refs,path])
    #print(ret)
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
    text = re.sub(r"\(\S+\)", " ", text)
    text = re.sub(r"may refer to:|You can help Wikipedia by expanding it .", " ", text)
    text = re.sub(r"\[citation needed\]|\[edit\]|\[ edit\ ]", " ", text)
    text = re.sub(r"url\(.+\)", " ", text)
    text = re.sub(r" +", " ", text)
    
    return text


def get_all_nodes(node):
    #print(node.title)
    nodes = [node]
    for child in node.children:
        if child.title != None:
            nodes.extend(get_all_nodes(child))
    return nodes

def process_nodes(nodes):
    overalldict = []
    path_list = []

    for node in nodes:
        flag = True
        #print("--------------")
        #print(node.title)
        #print(node.passages)
        keywords_list = []
        temp_node = node
        while temp_node:
            keywords_list = [temp_node.title] + keywords_list
            temp_node = temp_node.parent
        path_list.append(node.title)       
        dict = {}
        dict["keywords"] = list(keywords_list)
        dict["path"] = list(path_list)
        dict["passages"] = ""
        
        if hasattr(node, 'children') and not node.children:
            if node.passages == None:
                continue
            match = re.search(r'(.*?)<h', node.passages)
            if match:
                dict["passages"] = match.group(1)
            else:
                dict["passages"] = node.passages
        
        else:
            
            for child in node.children:
                if child.level == -1:
                    answer = child.passages
                    if answer == None:
                        flag = False
                        break
                    match = re.search(r'(.*?)<h', answer)
                    if match:
                        dict["passages"] = match.group(1)
                    else:
                        
                        dict["passages"] = answer
                    break


        #print(dict)
        if flag:
            overalldict.append(dict)
        #print("--------------")
        #print(dict)
    return overalldict
    



def add_node(root,sections,headings):
    level = root.level + 1
    
    
    for i, section in enumerate(sections):
        if i == 0:
            temp = MyClass(None,section,-1,parent=root)
        else:
            temp = MyClass(headings[i-1],section,level,parent=root)


def find_hi(i_level,text):
    #print("-----------------")
    
    #print("i: ", i_level)
    pattern_start = f"<h{i_level}>"
    pattern_end = f"</h{i_level}>"
    spans = [obj for obj in re.finditer(pattern_start, text)]
    #如果一个没找到，直接返回
    if not spans:
        return [],[]
    spans_end = [obj for obj in re.finditer(pattern_end, text)]
    all_sections = []  # n+1
    all_headings = []  # n
    if len(spans) != len(spans_end):  # if not matching, return lead section
        potential_end_idx = len(text)
        if spans:
            potential_end_idx = min(potential_end_idx, spans[0].start())
        if spans_end:
            potential_end_idx = min(potential_end_idx, spans_end[0].start())
        return [text[:potential_end_idx]], []

    last_section_start_idx = 0
    for i in range(len(spans)):
        left_tag_start_idx, left_tag_end_idx = spans[i].start(), spans[i].end()
        right_tag_start_idx, right_tag_end_idx = spans_end[i].start(), spans_end[i].end()
        last_section = text[last_section_start_idx:left_tag_start_idx] # get last section
        last_section_start_idx = right_tag_end_idx + 1
        all_sections.append(last_section)
        if len(last_section) < 10 and i != 0: # if last section is too short, skip last section
            all_headings = all_headings[:-1]
            all_sections = all_sections[:-1]
        cur_heading = text[left_tag_end_idx:right_tag_start_idx]
        #print("cur_heading")
        #print(cur_heading)
        #print(len(cur_heading.split(" "))>10)
        #print(all_sections)
        #print(all_headings)
        
        if "\n" in cur_heading or len(cur_heading.split(" "))>10: # return all sections before current heading
            #print("????")
            #print(all_sections)
            #print(all_headings)
            return all_sections, all_headings
        all_headings.append(cur_heading.lower().strip())
    last_section=text[last_section_start_idx:]
    all_sections.append(last_section)

    if len(all_sections) != 1 and len(last_section) < 10: # if last section is too short, skip last section
            all_headings = all_headings[:-1]
            all_sections = all_sections[:-1]
    #if len(all_headings) >= 10: # if too many headings, return only lead section
        #return all_sections[:1], []
    assert len(all_sections) == len(all_headings) + 1, text
    #print(all_sections)
    #print(all_headings)
    return all_sections, all_headings


def build_tree(root):
    #print("-----------")
    #print("root.title:",root.title)
    #print("root.level:",root.level)
    text = root.passages
    level = root.level + 1
    if level > 7:
        return root
   
    sections, headings = find_hi(level,text)
    #print("sections: ", sections)
    #print("headings: ", headings)
    if sections == []:
        return  root
    elif len(sections) == 1 and sections[0] == "":
        root.passages = ""
        return root
    else:
        add_node(root,sections,headings)
        for node in PreOrderIter(root):
            if node.is_leaf and node.title != None:
                build_tree(node)


#def update_answer(node):






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
                url, keywords, passages, cur_refs,path = x
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

if __name__ == '__main__':
    main()
