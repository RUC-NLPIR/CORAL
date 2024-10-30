from IPython import embed
import os
import json
import time
import argparse
from tqdm import tqdm, trange
from chat_promptor import ConversationPromptor
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
    promptor = ConversationPromptor(enable_demo=args.enable_demo,demo_file=args.demo_file_path)
    model_kwargs = {"temperature": 0.7}
    api_key = OPENAI_KEYS[args.open_ai_key_id]
    # print(args.base_url)
    generator = ChatGenerator(api_key, args.base_url,args.n_generation, **model_kwargs)
    
    
    
    # test_dataset    
    output_file_path = os.path.join(args.work_dir, "conversational_query.json")
    # has_qrel_labels_samples = get_has_qrel_label_sample_ids(args.qrel_file_path)
    

    begin_time = time.time()
    
    # predict
    
    temp = 0
    responses = {}
    with open(output_file_path, "w") as fw:
        for line in tqdm(open(args.test_file_path), "r"):
            data = json.loads(line)
            id = data["id"]
            result = id[2:]
            sample_id, turn_id = map(int,result.split("_"))
            if sample_id != temp:
                temp = sample_id
                if sample_id >= 2:
                    prompt = promptor.build_prompt(given_text)
                    #print(prompt)
                    #print(len(prompt.split(" ")))

                    try:
                        id_list,conversational_question_list = generator.generate(prompt, promptor.parse_returned_text)
                        #print(id_list,conversational_question_list)
                    except ValueError as e:
                        print(e)
                        
                    for i in range(len(id_list[0])):
                        record = {}
                        temp_turn_id = i+1
                        
                        temp_sample_id = sample_id - 1
                        id = "a_"+str(temp_sample_id)+"_"+str(temp_turn_id)
                        record['id'] = id
                        record["conversational question"] = conversational_question_list[0][i]
                        
                        #print(record)
                        fw.write(json.dumps(record))
                        fw.write('\n')
                        fw.flush()
                        
                
                given_text = "Topic: {}\n\n".format(data["keywords"][0])

            question = data["question"]
            response = data["response"]
            given_text += "{{Id:{}\nQuestion:{}\nResponse:{}}}\n\n".format(turn_id, question, response)
        
        prompt = promptor.build_prompt(given_text)
        #print(prompt)
        #print(len(prompt.split(" ")))

        try:
            id_list,conversational_question_list = generator.generate(prompt, promptor.parse_returned_text)
            #print(id_list,conversational_question_list)
        except ValueError as e:
            print(e)
                        
        for i in range(len(id_list[0])):
            record = {}
            temp_turn_id = i+1
            
            temp_sample_id = sample_id
            result_id = "a_"+str(temp_sample_id)+"_"+str(temp_turn_id)
            record['id'] = result_id
            record["conversational question"] = conversational_question_list[0][i]
                        
            #print(record)
            fw.write(json.dumps(record))
            fw.write('\n')
            

    
    print("{} Generation ok!, time cost {}".format(args.work_dir, time.time() - begin_time))



if __name__ == '__main__':
    main()
