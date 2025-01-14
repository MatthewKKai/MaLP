import evaluate
import json
from tqdm import tqdm
from model.utils import ChatGPTWrapper

engine = ChatGPTWrapper()
with open(r"results/dialogues_4_600.json", encoding="utf-8") as f:
    data = json.load(f)

def cal_rouge(predictions, references):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    return results

def cal_accuracy(predcitions, references):
    accuracy = evaluate.load('accuracy')
    results = accuracy.compute(predcitions=predictions, references=references)
    return results

def win_rate():
    pass

answers = []
with tqdm(total=100) as pbar:
    pbar.set_description(r"Response Generating")
    for i in range(100):
        query = "given the following dialogue: {0}, please mimic the Assistant to give a response regarding user's new question 'What should I do to ease my pain?'".format(str(data[i]))
        messages = [{'role':'user', 'content':query}]
        answer = engine.obtain_answer(messages)
        tmp = {i:answer}
        answers.append(tmp)
        pbar.update(1)

with open(r"resposne.json", "w", encoding="utf-8") as f:
    json.dump(answers, f)
# print(data[0])
