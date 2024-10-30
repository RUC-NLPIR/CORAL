import time
import openai
import httpx
from openai import OpenAI
from IPython import embed

# TODO: Write your OpenAI API here.
OPENAI_KEYS = []



# from https://github.com/texttron/hyde/blob/main/src/hyde/generator.py
class ChatGenerator:
    def __init__(self, 
                 api_key,
                 base_url,
                 n_generation,
                 **kwargs):
        self.model_name = 'gpt-4-turbo-2024-04-09'
        self.n_generation = n_generation
        self.kwargs = kwargs
        self.client = OpenAI(api_key=api_key,
                             base_url=base_url,
                             http_client=httpx.Client(
                                base_url=base_url,
                                follow_redirects=True,
                                ),
                            )
        #print(base_url)
    
    def parse_result(self, result, parse_fn):
        #print(result)
        choices = result.choices
        n_fail = 0
        res_id = []
        res_question = []
        
        for i in range(len(choices)):
            output = choices[i].message.content
            #print(output)
            id_list,conversational_question_list = parse_fn(output)
            
            #print(output)
            if not id_list:
                n_fail += 1
            else:
                res_id.append(id_list)
                res_question.append(conversational_question_list)
                
        return n_fail, res_id,res_question
                
        
    def generate(self, prompt, parse_fn):
        n_generation = self.n_generation
        n_try = 0
        # embed()
        # input()
        while True:
            if n_try == 5:
                raise ValueError("Have tried 5 times but still only got 0 successful outputs")
                
            while True:
                try:
                    result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "{}".format(prompt)},
                        ],
                        n=n_generation,
                        **self.kwargs
                    )
                    #print(result)
                    # embed()
                    # input()
                    break
                except openai.RateLimitError:
                    time.sleep(20)
                    print("Trigger RateLimitError, wait 20s...")
                except Exception as e:
                    # 处理其他异常情况
                    time.sleep(20)
                    print("Exception:", e)

            n_fail, res_id,res_question = self.parse_result(result, parse_fn)
            #print(n_fail, res_id,res_question)

            if n_fail == 0:
                return res_id,res_question

                
            n_try += 1
    
        


