from IPython import embed
import os
import json
import time
import argparse
from tqdm import tqdm, trange
from chat_promptor import QueryPromptor
from generator import ChatGenerator, OPENAI_KEYS
from utils import set_seed, get_finished_sample_ids, get_has_qrel_label_sample_ids


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path", type=str, required=True)
    parser.add_argument("--demo_file_path", type=str, required=True)
    # parser.add_argument("--qrel_file_path", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True, help='output query path.')
    parser.add_argument("--n_generation", type=int, required=True, help='the number for generation')
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--open_ai_key_id", type=int, choices=[0,1,2,3,4,5], required=True)
    parser.add_argument("--base_url", type=str, default=None,help = "paid domain name")
    parser.add_argument("--enable_demo", action="store_true")
    parser.add_argument("--enable_cot", action="store_true")
    
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "parameters.txt"), "w") as f:
        params = vars(args)
        f.write(json.dumps(params, indent=4))
        
    return args


def main():
    args = get_args()    
    set_seed(args) 
    
    # model and promptor setting
    promptor = QueryPromptor(enable_demo=args.enable_demo,demo_file=args.demo_file_path,enable_cot=args.enable_cot)
    model_kwargs = {"temperature": 0.7, "max_tokens": 64, "stop": promptor.stop_tokens}
    api_key = OPENAI_KEYS[args.open_ai_key_id]
    # print(args.base_url)
    generator = ChatGenerator(api_key, args.base_url,args.n_generation, **model_kwargs)
    
    
    
    # test_dataset    
    output_file_path = os.path.join(args.work_dir, "generate_query.json")
    failure_file_path = os.path.join(args.work_dir, "failure_id.json")
    finished_samples = get_finished_sample_ids(output_file_path)
    # has_qrel_labels_samples = get_has_qrel_label_sample_ids(args.qrel_file_path)
    

    begin_time = time.time()
    
    failure_id = []
    # predict
    with open(output_file_path, "a+") as fw:
        for line in tqdm(open(args.test_file_path), "r"):
            data = json.loads(line)
            sample_id = data["id"]
            if sample_id in finished_samples:
                continue
            keyword_list = data["keywords"]
            passage_list = data["passages"]

            
            keyword_text = ", ".join(keyword_list)  # 使用逗号将关键字连接起来
            
            keyword_text += "\n\n"  # 添加换行符
            keyword_text = "# Keywords: " + keyword_text  # 添加关键字标签

            passage_text = "# Response: "

            for pas in passage_list:
                if len(pas) == 2 and pas[1] == '[0]':
                    pas_text = pas[0]
                else:
                    pas_text = "".join(pas)
                passage_text += pas_text    
            
            
            given_text = keyword_text + passage_text
            prompt = promptor.build_prompt(given_text)
            print(prompt)
            # print(len(prompt.split(" ")))
            try:
                query_list = generator.generate(prompt, promptor.parse_returned_text)
            except ValueError as e:
                failure_id.append(sample_id)
                print(e)
                query_list = generator.generate(prompt, promptor.parse_returned_failure_text)

            record = {}
            record['sample_id'] = data["id"]
            record['predicted_query'] = query_list
              
            fw.write(json.dumps(record))
            fw.write('\n')
            fw.flush()

    with open(failure_file_path, "w") as f:
        f.write(json.dumps(failure_id))
    print("{} Generation ok!, time cost {}".format(args.work_dir, time.time() - begin_time))


if __name__ == '__main__':
    main()
