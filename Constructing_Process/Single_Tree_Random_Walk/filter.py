import os
import json



file_path = ''
write_file_path = ""

a_keywords_path = ""
b_keywords_path = ""

with open(a_keywords_path, 'r') as f:
    a_keywords = json.load(f)
with open(b_keywords_path, 'r') as f:
    b_keywords = json.load(f)
ab_keywords = a_keywords + b_keywords
print(len(ab_keywords))

def filter_data(dict):
# 这里定义筛选逻辑
# 要求每个sample的turn数大于等于4
    turns = []
    turn_id = 1
    refs_count = 0
    for d in dict:
        if d["text"] != []:
            temp = {}
            temp["turn_id"] = turn_id
            temp["keywords"] = d["title"]
            temp["text"] = d["text"]
            temp["refs"] = d["references"]
            if d["references"] != []:
                refs_count += 1
            turn_id += 1
            turns.append(temp)
    if turn_id >=12 and turn_id <= 21 and refs_count >= (turn_id - 1)/2 + 1:
       return turns
    else:
       return []




# 存储筛选后的数据
filtered_data = []

sample_id = 1601
# 遍历文件夹



with open(file_path, 'r') as f:
        
    overdict = []  
    keys = ""
    url = ""
    for line in f:
        dict = json.loads(line)
        temp =dict["title"]
        #print(temp)
        if temp[0] in ab_keywords:
            continue   
        #print(dict["title"])
        if temp[0] != keys:
            keys = temp[0]
            
            result = filter_data(overdict)  
            overdict = []
            if result != []:
                filtered_dict = {}
                filtered_dict["sample_id"] = sample_id
                filtered_dict["turns"] = result
                filtered_dict["url"] = url      
                sample_id += 1
                filtered_data.append(filtered_dict)

        overdict.append(dict)  
        url = dict["url"]     
    print(sample_id)     

            
    

# 将筛选后的数据写入新的JSON文件
with open(write_file_path, 'w') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
    