from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

default_message = "Don't know"

def load_data(n_samples = 1505):
    df1 = pd.read_csv("bots/data/S08_question_answer_pairs.txt", sep = '\t')
    df2 = pd.read_csv("bots/data/S09_question_answer_pairs.txt", sep = '\t')
    df3 = pd.read_csv("bots/data/S10_question_answer_pairs.txt", sep = '\t', encoding = 'ISO-8859-1')
    data = pd.concat([df1, df2, df3], ignore_index= True)
    data = data.dropna()
    data = data.drop_duplicates(subset='Question')
    # data = data.sample(n_samples, ignore_index= True)
    data = data[['Question', 'Answer']]
    print(f"----------------loaded {data.shape} of data")
    return data

def load_model(model_name = 'paraphrase-albert-small-v2'):
    model = SentenceTransformer(model_name)
    print('----------------model loaded')
    return model

def embeddings_data(data: pd.DataFrame, model, load_file = True):
    if load_file:
        data_embeddings = np.load("bots/embedded-qa.npy")
    else:
        questionsData = data.Question.to_list()
        data_embeddings = model.encode(questionsData)
        np.save("chatapp/chatbot/embedded-qa.npy", data_embeddings)
    print('----------------data embedded', data_embeddings.shape)
    return data_embeddings

def getAnswer(question, data, data_embeddings, model):
    question_embedding = model.encode(question)

    similarity_score = cosine_similarity(
        [question_embedding],
        data_embeddings
    ).flatten()

    max_score_index = np.argmax(similarity_score)
    print("----------------found answer at: ", max_score_index)
    score = similarity_score[max_score_index]
    answer = data.iloc[max_score_index, 1]
    similarityQuestion = data.iloc[max_score_index, 0]

    _resultdata = {'Question': question, 'Simiarity Q': similarityQuestion, 'Score': score, 'Answer': answer}
    return _resultdata

# data = load_data()
# model = load_model()
# data_embeddings = embeddings_data(data, model)
# print(getAnswer("What serves as the capital of Indonesia?", data, data_embeddings, model))
# print(getAnswer("", data, data_embeddings, model))