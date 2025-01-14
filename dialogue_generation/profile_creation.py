import sys
import os
# print(sys.path)
# print(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.getcwd()))
from utils import ChatGPTWrapper
from prompts import profile_generation_prompts, dialogue_preference
import json
import re
from tqdm import tqdm



# pattern = r'{.*}'
profilePrompt = profile_generation_prompts
diaPrefer = dialogue_preference
path = r"data/raw_data/Medical_Dialogue"

class profile_creation():
    def __init__(self, engine, role, preference):
        self.engine = engine
        self.role = role
        self.preference = preference
    
    def obtain_profile(self, description):
        prompt = profilePrompt["profile_creation"].format(self.role, self.preference, description)
        messages = [{'role':'user', 'content':prompt}]
        # print("This is the prompt:{0}".format(prompt))
        answer = self.engine.obtain_answer(messages)
        # print("This is the answer:{0}".format(answer))
        match = re.search(r"{.*}", str(answer))  # apply regex to get the json text
        if match is None:
            profile = answer
        else:
            profile = match.group()
        # print(profile)
        return profile

    def raw_description(self, path):
        descriptions = []
        # print(os.getcwd())
        with open(path, encoding="utf-8") as f:
            data = f.readlines()
        # length can be replaced by the len(data), too long for now
        for i in range(30000):
            if data[i]=='Patient:\n':
                descriptions.append(data[i+1])
        return descriptions
    
engine = ChatGPTWrapper()
profile_generator = profile_creation(engine, "patient", diaPrefer)
descriptions = profile_generator.raw_description(os.path.join(path, "healthcaremagic_dialogue_4.txt"))
profiles = []
with tqdm(total=len(descriptions)) as pbar:
    pbar.set_description(r"Profile generating:")
    for description in descriptions:
        profile = profile_generator.obtain_profile(description)
        profiles.append(profile)
        pbar.update(1)

# print(profiles[1:10])

with open(r"profiles_4.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f)
