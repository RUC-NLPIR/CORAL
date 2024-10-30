from IPython import embed
import re
import json


def check_length(prompt, max_length):
    n = len(prompt.split(' '))
    if n >= max_length:
        return False
    return True


class QueryPromptor:
    def __init__(self, enable_demo=False, demo_file=None, enable_cot=False) -> None:    
        
        self.instruction = "Given the keyword chain of the question and its response, generate the original question. If the response is not informative enough to help you reconstruct the question, please rely on the provided keyword chain to generate the question. The keyword chain consists of terms where each term is a more specific or detailed subset of the previous one, with the last term being the most specific or important. The question you generate should focus on the last keyword in the chain and include it explicitly."
        self.enable_cot = enable_cot
        self.middle_instruction = "Given the following keyword chain and response: "

        if enable_demo == True:
            self.demo = self.get_demo(demo_file)
            if self.demo != "":
                self.instruction += " I will give you several example dialogs, where each example contains several keywords, a response, and a question that you need to generate."
                if enable_cot:
                    self.instruction += " The generating query part begins with a sentence explaining the reason for the generated question."
        if enable_cot:
            self.tail_instruction = "Now, you should give me the original question given the keywords and its response. The output format should always be: \"Question: $Question. So the question should be written as: $Question.\" Note that you should always try to generate it. Never ask for clarification or say you don't understand it in the generated question. Go ahead!"
        else:
            self.tail_instruction = "Now, you should give me the original question given the keyword chain and its response. The output format should always be: Question: $Question. Note that you should always try to generate it. Never ask for clarification or say you don't understand it in the generated question. Go ahead!"
        self.stop_tokens = ['\n']
    
    def get_demo(self, demo_file):
        try:
            demos = []
            with open(demo_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    demos.append(data)
        except:
            print("warning: No demonstration file.")
            return ""
        
        examples = []
        for demo in demos:
            keyword_text = ", ".join(demo["keywords"])
            demo_text = "Keywords: {}\nResponse: {}Question: {}".format(keyword_text, demo['response'],demo['question'])         
            examples.append(demo_text)
        
        for i in range(len(examples)):
            examples[i] = "Example #{}:\n".format(i+1) + examples[i]
        
        return "\n\n".join(examples)
              
    def build_prompt(self, example):
        this_example = example
        this_prompt = [self.instruction, self.middle_instruction, this_example, self.tail_instruction]
        this_prompt = "\n\n".join(this_prompt)
        return this_prompt

        
    
    def parse_returned_text(self, text):
        text = text.strip()
        
        if text[:9] != "Question:":
            # print("???????????")
            return None
        if not self.enable_cot:
            # print(text[9:])
            return text[9:]
        else:
            fixed_sentence = "So the question should be written as:"
            index = text.find(fixed_sentence)
            if index != -1:
                cot = text[:index]
                question = text[index + len(fixed_sentence):]
                return [cot.strip(), question.strip()]
            else:
                return None
            
    def parse_returned_failure_text(self, text):
    # parse_returned_text 出错的原因大概输入文本太长。
        text = text.strip()
        return text