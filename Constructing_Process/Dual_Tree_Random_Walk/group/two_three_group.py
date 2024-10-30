import json

file_path = ""
write_file_path = ""

with open(file_path, "r") as f:
    data = json.load(f)
    used_ids = set()
    result = []
    
    for sample_id, related_ids in data.items():
        sample_id_int = int(sample_id)
        if sample_id_int in used_ids:
            continue
        used_ids.add(sample_id_int)
        for related_id in related_ids:
            
            if related_id not in used_ids:
                used_ids.add(related_id)
                result.append(tuple([sample_id_int, related_id]))
                break
    
with open(write_file_path, "w") as wf:
    json.dump(result, wf, indent=4)