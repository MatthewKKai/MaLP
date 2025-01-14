import requests
import json
import torch
from torch import nn

url = r"Your own chatrobot api here"
header = {"Authorization": "Your own api key here",
        "content-type": "application/json"}

# should be json format, one-time call for one query
prompt = "{0}, {1}".format()
data = json.dumps({"query":"Can magnets pick up a penny?",
                   "conversation_id":"a3eedc5a-7dd8-49d5-8024-8be023f625b1"})

# r = requests.request("POST", url, headers=header, data=data)
# print(r.text)

class ChatGPTWrapper():
    def __init__(self, url: str=r"Your own chatrobot api here", headers: dict={"Authorization": "Your own api key here", 
    "content-type": "application/json"}, model: str="gpt-3.5-turbo-0301"):
        self.url = url
        self.headers = headers
        self.isMulti = False # denotes if this is multi-round conversation
        self.model = model

    def obtain_response(self, messages):
        playload = self.obtain_playload(messages)
        try:
            response = requests.request("POST", self.url, headers=self.headers, data=playload)
            if response.status_code==200:
                return response
            else:
                return self.obain_repsone(messages)
        except Exception as e:
            return r'{"Error":"Connection Error, please check your network and settings"}'

    def obtain_answer(self, messages):
        # r could be str due to exceptions
        r = self.obtain_response(messages)
        if isinstance(r, str):
            return "Error Networking, check your network"
        else:
            text = json.loads(r.text)
            # print(text["data"]["messages"][0]["content"])
            try:
                answer = text["data"]["response"][0]["content"]
            except Exception as e:
                answer = str(text)
            return answer

    def obtain_playload(self, messages):
        # placeholder, further parse input data
        if isinstance(messages, list):
            # print("Reach Here 1")
            playload = json.dumps({'model': self.model, 'messages': messages})
        else:
            print("Please transform the messages into list form")
            playload = json.dumps({'model': self.model, 'messages': "Invalid messages"})
        return playload

    # convId no longer avaliable
    # def obtain_convId(self):
    #     return self.convIds


# rewrite prompts
class ReWriter(nn.Module):
    def __init__(self, engine):
        self.engine = engine
        self.prompt = r"Please Rewrite this question in terms of the same meaning"

    def rewrite(self, x):
        text = self.prompt+":"+x
        rewritten_text = self.engine.obtain_answer(text)
        return rewritten_text

# judge if usable
class Identifier(nn.Module):
    def __init__(self, engine):
        self.engine = engine
        self.prompt = r"Please check if these two phrase share the same meaning, answer 'Yes' or 'No' only"

    def check_answer(self, x1, x2):
        text = self.prompt + ":" + x1 + ";" + x2
        answer = self.engine.obtain_answer(text)
        return answer

# summarize the learned knowledge
class Summarizer():
    def __init__(self, engine):
        self.engine = engine
        self.prompt = r"Please list the common-sense knowledge and user-specific knowledge (including user dialogue preference) item by item."

    def summarize(self):
        summarization = self.engine.obtain_answer(self.prompt)
        return summarization

# cgw = ChatGPTWrapper()   
# print("---------------Original Generation-------------")
# print("Question: Can magnets pick up a penny?")
# text = cgw.obtain_answer(data)
# print(text)
# print("---------------Generation with Prompt-------------")
# x_with_prompt = json.dumps({"query":"Can magnets pick up a penny? Assume the penny is made of copper, copper is not magnetic."})
# text = cgw.obtain_answer(x_with_prompt)
# print(text)
# print("---------------Regenerate with Rewritten Input-------------")
# rewriter = ReWriter(cgw)
# identifier = Identifier(cgw)
# rewritten_text = rewriter.rewrite("Can magnets pick up a penny? Assume the penny is made of copper, copper is not magnetic.")
# print("rewritten text: {0}".format(rewritten_text))
# is_same = identifier.check_answer("Can magnets pick up a penny? Penny is made of copper, copper is not magnetic", rewritten_text)
# print("Is the rewritten text sharing the same meaning as previous?---{0}".format(is_same))
# x_rewrite = json.dumps({"query":rewritten_text})
# text = cgw.obtain_answer(x_rewrite)
# print(text)
