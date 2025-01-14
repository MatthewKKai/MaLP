from utils import ChatGPTWrapper
from prompts import conversation_prompts, profile_based_conv_prompts
from random import randint
from tqdm import tqdm
import json
import time

personality_list = ['angry', 'sad', 'happy', 'shy']
occupation_tpoic = {'student':'return a book', 'lawer':'pick up lawsuit documents', 'programmer':'consult technical demands'}

"""
Define a character with peronality, task scenario etc.
"""
# class character():
#     def __init__(self, engine, 
#                 prompt, 
#                 occupation: str="customer", 
#                 topic: str="return a book", 
#                 personality: str="angry"):
#         self.engine = engine
#         self.prompts = prompt
#         self.occupation = occupation
#         self.personality = personality
#         self.topic = topic

#     def start(self):
#         initialization = self.prompts["Start"].format(self.occupation, self.personality, self.topic)
#         start = self.engine.obtain_answer(initialization)
#         convId = self.engine.convId
#         return start, convId

#     def not_start(self, x):
#         initialization = self.prompts["Not_Start"].format(self.occupation, self.personality, self.topic, x)
#         print(initialization)
#         start = self.engine.obtain_answer(initialization)
#         convId = self.engine.convId
#         return start, convId

#     def respond(self, x):
#         return self.engine.obtain_answer(x)

'''
old version conversation no longer usable since the conversation id changed
'''
# class conversation():
#     def __init__(self, person, agent, max_round: int=10):
#         self.person = person
#         self.agent = agent
#         self.max_round = max_round

#     def conv(self):
#         diologue_record = []
#         round = 0
#         # randomly initialize
#         if randint(0,1)==1:
#             who_start = self.person
#             utterance_p, convId_p = who_start.start()
#             utterance_a, convId_a = self.agent.not_start(utterance_p)
#             diologue_record.append({"start":{"person":utterance_p, "agent":utterance_a}})
#             print(diologue_record)
#             while round<self.max_round:
#                 if round==self.max_round-1:
#                     utterance_p = self.person.respond(utterance_a)
#                     utterance_p = utterance_p + self.agent.prompts["End"]
#                     utterance_a = self.agent.respond(utterance_p)
#                     diologue_record.append({round:{"person":utterance_p, "agent":utterance_a}})
#                     round+=1
#                 else:
#                     utterance_p = self.person.respond(utterance_a)
#                     utterance_a = self.agent.respond(utterance_p)
#                     diologue_record.append({round:{"person":utterance_p, "agent":utterance_a}})
#                     round+=1
#             return diologue_record
#         else:
#             who_start = self.agent
#             utterance_a, convId_a = who_start.start()
#             utterance_p, convId_p = self.person.not_start(utterance_a)
#             diologue_record.append({"start":{"person":utterance_p, "agent":utterance_a}})
#             print(diologue_record)
#             while round<self.max_round:
#                 if round==self.max_round-1:
#                     utterance_a = self.person.respond(utterance_p)
#                     utterance_a = utterance_a + self.agent.prompts["End"]
#                     utterance_p = self.agent.respond(utterance_a)
#                     diologue_record.append({round:{"person":utterance_p, "agent":utterance_a}})
#                     round+=1
#                 else:
#                     utterance_a = self.person.respond(utterance_p)
#                     utterance_p = self.agent.respond(utterance_a)
#                     diologue_record.append({round:{"person":utterance_p, "agent":utterance_a}})
#                     round+=1
#             return diologue_record

'''
Profile based character
'''
class character_w_profile():
    def __init__(self, engine, 
                character,
                profile, 
                prompt):
        self.engine = engine
        self.character = character # 'user'/'assistant' - different from the role field in the next following functions, denotes the defined charcter
        self.profile = profile
        self.profile_based_prompt = prompt
        self.messages = []

    def start(self, start):
        content = self.profile_based_prompt[start].format(self.character, self.profile)
        message = {'content':content, 'role': 'user'}
        self.messages.append(message)
        start_msg = self.engine.obtain_answer(self.messages)
        start_msg_tmp = {'content':start_msg, 'role': 'assistant'}
        self.messages.append(start_msg_tmp)
        # self.msg_history()
        return start_msg

    def respond(self, content):
        # for multi-conv, need to append all dialogue history to messages
         message = {'content': content, 'role': 'user'}
         self.messages.append(message)
         response = self.engine.obtain_answer(self.messages)
         self.messages.append({'content': response, 'role': 'assistant'})
         return response

    def msg_history(self):
        print(self.messages)

'''
New Version Conversation: Update Multi-Conv generation
'''
class conv_w_profile():
    def __init__(self, user,
                assistant,
                max_round: int=6):
        self.user = user
        self.assistant = assistant
        self.max_round = max_round

    def conv(self):
        round = 0
        dialogue_record = []
        # initilization
        utterance_a = self.assistant.start("Assistant_Start")
        utterance_u = self.user.start("User_Start")
        dialogue_record.append({round:{'Assistant': utterance_a, 'User': utterance_u}})
        round+=1
        while round<self.max_round:
            if round==self.max_round-1:
                utterance_a = utterance_a + self.assistant.profile_based_prompt["End"]
                utterance_u = self.user.respond(utterance_a)
		# time.sleep(5)
                utterance_a = self.assistant.respond(utterance_u)
                dialogue_record.append({round:{'Assistant':utterance_a, 'User':utterance_u}})
                round+=1
            else:
                utterance_a = self.assistant.respond(utterance_u)
		# time.sleep(5)
                utterance_u = self.user.respond(utterance_a)
                dialogue_record.append({round:{'Assistant':utterance_a, 'User':utterance_u}})
                round+=1
        # self.user.msg_history()
        # self.assistant.msg_history()
        return dialogue_record


# define engine for person
engine1 = ChatGPTWrapper()
# define engine for agent
engine2 = ChatGPTWrapper()


with open(r"profiles_4.json", encoding="utf-8") as f:
    profiles = json.load(f)

all_dialogue = []
with tqdm(total=len(profiles)) as pbar:
    pbar.set_description(r"Dialogue Generating")
    for i in range(0, len(profiles)):
        person = character_w_profile(engine1, character="patient", profile=profiles[i], prompt=profile_based_conv_prompts)
        assistant = character_w_profile(engine2, character="doctor", profile="patient", prompt=profile_based_conv_prompts)
        conversation_test = conv_w_profile(person, assistant)
        dialogue = conversation_test.conv()
        all_dialogue.append(dialogue)
        pbar.update(1)

with open(r"results/dialogues_4.json", "w", encoding="utf-8") as f:
    json.dump(all_dialogue, f)