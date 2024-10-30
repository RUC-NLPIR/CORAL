import json
file_path = ""
write_file_path = ""



# 读取 JSON 文件
with open(file_path, 'r') as file,open(write_file_path, "w") as write_file:
    data = json.load(file)

    # 确保 data 是一个列表
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # 对每个字典进行处理
                #print(item)
                sample_id = item['sample_id']
                turns = item['turns']
                for turn in turns:
                    new_dict = {}
                    turn_id = turn['turn_id']
                    keywords = turn['keywords']
                    text = turn['text']
                    refs = turn['refs']
                    new_dict['id'] = "a_"+"{}_{}".format(sample_id, turn_id)
                    new_dict['keywords'] = keywords
                    first_elements = [sublist[0] for sublist in text]

                    new_dict['passages'] = ' '.join(first_elements)
                    new_dict['refs'] = refs
                    new_dict["url"] = item["url"]
                    write_file.write(json.dumps(new_dict, ensure_ascii=False) + '\n')

                
            else:
                print("列表中的元素不是字典：", item)
    else:
        print("JSON 数据不是一个列表")