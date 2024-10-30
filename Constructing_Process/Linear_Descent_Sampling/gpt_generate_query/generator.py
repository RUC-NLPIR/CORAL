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
    
    def parse_result(self, result, parse_fn):
        # print(result)
        choices = result.choices
        n_fail = 0
        res = []
        
        for i in range(len(choices)):
            output = choices[i].message.content
            # print(output)
            output = parse_fn(output)
            
            #print(output)
            if not output:
                n_fail += 1
            else:
                res.append(output)
                
        return n_fail, res
                
        
    def generate(self, prompt, parse_fn):
        n_generation = self.n_generation
        output = []
        n_try = 0
        # embed()
        # input()
        while True:
            if n_try == 5:
                if len(output) == 0:
                    raise ValueError("Have tried 5 times but still only got 0 successful outputs")
                output += output[:5-len(output)]
                return output
                break
            
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

            n_fail, res = self.parse_result(result, parse_fn)
            output += res
            
            if n_fail == 0:
                return output 
            else:
                n_generation = n_fail
                
            n_try += 1
    
        


