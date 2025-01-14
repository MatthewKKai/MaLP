conversation_prompts = {
    'Start': "Next, we will simulate a conversation that means you say one utterance, I say one utterance. Now you are a {0} who is easily getting {1} and you are about to {2}. Remember Do Not generate the whole diologue, only your part. You start first.",
    'Not_Start': "Next, we will simulate a conversation that means you say one utterance, I say one utterance. Now you are a {0} who is easily getting {1} and you are about to {2}. Remember Do Not generate the whole diologue, only your part. I start first:{3}",
    'End':'(It is time to end this conversation)'
}

profile_generation_prompts = {
    'profile_creation':'Please create a {0} profile in text json file format of three fields ("personal_information", "symptoms", "dialogue_preference"), "dialogue_preference" field is randomly selected from {1} and other fields consist of a sentence concluded from this description:{2}'
}
dialogue_preference = ["prefer concise answer", "prefer detailed description", "prefer polite response"]

# 'Assistant' and 'User' are different from the api role field, denotes character (e.g., {0})
profile_based_conv_prompts = {
    'Assistant_Start':'Next we will simulate a conversation that means you generate only one utterance per time. Now assume you are a {0} and aims to provide professional advices to {1}, you start first',
    'User_Start':'Next we will simulate a conversation that means you say one utterance, I say one utterance. Now assume you are a {0} asking advices from me, this is your profile:"{1}", I start first:"Hi, how can I help you"',
    'End':'(It is time to end this conversation)'
}

# evaluation prompts
evaluation = {
    'Q1': "What's the user's personal profile?",
    'Q2': "What's the common-sense knowledge?",
    'Q3': "What's the user-specific knowledge?",
    'Resposne': "I'm uncomfortable again in terms of my previous symptoms, can you also give me some advice?"
}

preference_selection = {"Query": "what's the dialogue preference of the given dialogue {0}, choose from three  ['prefer concise answer', 'prefer detailed description', 'prefer polite response'] options, do not generate other content."}