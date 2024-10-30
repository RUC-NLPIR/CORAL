import json
import copy
import random

file_path = ""
tuple_file_path = ""

write_file_path = ""

with open(file_path, "r") as f:
    merge_data = json.load(f)
    #print(merge_data[1495]["sample_id"])



def process_combinationn(dict1,dict2,sample_id):

    url = [dict1["url"],dict2["url"]]
    turns1 = dict1["turns"]
    turns2 = dict2["turns"]
    final_turns = []
    turn_id = 1
    for turn in turns1:
        if turn["text"] != [] and turn["refs"] != []:
            turn["turn_id"] = turn_id
            turn_id += 1
            final_turns.append(turn)
        random_choice = random.choice([7,8,9,10])
        if turn_id > random_choice:
            break
    for turn2 in turns2:
        if turn2["text"] != [] and turn2["refs"] != []:
            turn2["turn_id"] = turn_id
            turn_id += 1
            final_turns.append(turn2)  
        random_choice = random.choice([14,15,16,17,18,19]) 
        if turn_id >= random_choice:
            break
    return {"url":url,"turns":final_turns}
        





result = []
with open(tuple_file_path, "r") as f:
    data = json.load(f)
    count = 1601
    for d in data:
        part1_id = d[0] - 1
        part2_id = d[1] - 1
        part1_dict = merge_data[part1_id]
        #print(part2_id)
        part2_dict = merge_data[part2_id]
        
        my_dict = {}
        my_dict = process_combinationn(part1_dict,part2_dict,sample_id = count)
        
        #print(len(my_dict["turns"]))
        if len(my_dict["turns"]) >= 11 and len(my_dict["turns"]) <= 20:
            my_dict["sample_id"] = count
            
            count += 1
            # 需要深拷贝！
            result.append(copy.deepcopy(my_dict))
    print(count)


with open(write_file_path, "w") as wf:
    json.dump(result, wf, indent=4)
            

    