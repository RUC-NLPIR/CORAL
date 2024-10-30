from IPython import embed
import re
import json


def check_length(prompt, max_length):
    n = len(prompt.split(' '))
    if n >= max_length:
        return False
    return True


class ConversationPromptor:
    def __init__(self, enable_demo=False, demo_file=None, enable_cot=False) -> None:    
        
        self.instruction = "Given a topic and corresponding question and response pairs. The questions are arranged in a logical, progressively deeper sequence, where each subsequent question delves deeper into the topic based on the previous one. I would like you to convert the original question into a conversational form. The goal is to rewrite it without any grammatical errors while preserving its original intent as closely as possible. It is necessary to consider the omission and reference to the previous question and response in the generated conversational question.\n\n"
        self.enable_cot = enable_cot
        self.middle_instruction = "Now I will give you a topic and the corresponding question and response pairs:"

        if enable_demo == True:
            self.demo = self.get_demo(demo_file)
            if self.demo != "":
                self.instruction += "I will give you one example multi-turn dialog, where each turn contains a original question, a conversational question, a response, and the corresponding analysis.\nExample:\nTopic: depression\nConversations:\n"
                self.instruction += self.demo
        
        self.tail_instruction = "Please consider the question and response from the previous text when generating the current conversational question, but there is no need to generate the response. The output should be in the following format:\n\"Turn #{$turn_id}\nId:{$id}\nOriginal Question:{$original question}\nConversational Question:{$conversational question}\nReason:{$reason}\"\n\n"
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
        
        turns = []
        for i,demo in enumerate(demos):
            demo_text = "Turn #{}:\n".format(i+1)
            demo_text += "Original Question{}: {}\n".format(i+1, demo['original_question'])
            demo_text += "Conversational Question{}: {}\n".format(i+1, demo['conversational_question'])
            demo_text += "Response{}: {}\n".format(i+1, demo['response'])
            demo_text += "Analysis: {}".format(demo['analysis'])
             
            turns.append(demo_text)
        
        return "\n\n".join(turns)
              
    def build_prompt(self, example):
        this_example = example
        this_prompt = [self.instruction, self.middle_instruction, this_example, self.tail_instruction]
        this_prompt = "\n\n".join(this_prompt)
        return this_prompt

        
    
    def parse_returned_text(self, text):
        text = text.strip()
        
        id_list = []
        conversational_question_list = []
        for i in range(2,16):
            start_index = text.find(f"Turn #{i-1}")
            end_index = text.find(f"Turn #{i}")
            if end_index == -1:
                if start_index == -1:
                    break
                end_index = len(text)

            extracted_text = text[start_index:end_index]
            extracted_text = extracted_text.strip()

            # extract id
            substring_id = "Id:"
            substring_original_question = "Original Question:"
            start_id_index = extracted_text.find(substring_id)
            if start_id_index == -1:
                return None, None
            start_id_index += len(substring_id)
            end_id_index = extracted_text.find(substring_original_question)
            if end_id_index == -1:
                return None, None
            id_str = extracted_text[start_id_index:end_id_index].strip()
            id = int(id_str)
            id_list.append(id)

            # extract conversational question
            substring_conversational_question = "Conversational Question:"
            substring_reason = "Reason:"
            start_question_index = extracted_text.find(substring_conversational_question)
            if start_question_index == -1:
                return None, None
            start_question_index += len(substring_conversational_question)
            end_question_index = extracted_text.find(substring_reason)
            if end_question_index == -1:
                return None, None
            
            conversational_question_text = extracted_text[start_question_index:end_question_index].strip()
            conversational_question_list.append(conversational_question_text)
        
        if len(id_list) != len(conversational_question_list):
            return None, None
        
        #print(id_list, conversational_question_list)
        return id_list, conversational_question_list
        
            
    