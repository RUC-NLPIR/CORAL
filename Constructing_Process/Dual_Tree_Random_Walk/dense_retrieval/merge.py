import json

file_path = ""
write_file_path = ""

my_dict = {}
with open(file_path, "r") as f:
    for line in f:
        parts = line.split()
        sample_id = int(parts[0])
        #print(type(sample_id))
        retrieve_result = int(parts[2])
        if sample_id not in my_dict:
            my_dict[sample_id] = [retrieve_result]
        else:
            my_dict[sample_id].append(retrieve_result)
       
with open(write_file_path,"w") as wf:
    sorted_by_key = dict(sorted(my_dict.items()))
    json.dump(sorted_by_key, wf, indent=4)