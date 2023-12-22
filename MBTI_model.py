import keras;import numpy as np;
import sys;
from transformers import BertTokenizer
from huggingface_hub import from_pretrained_keras
import os;
from transformers import AutoTokenizer
# from sklearn.preprocessing import LabelBinarize
#  파일경로 : C:/Users/wjkim/TIKI/School/Machine/ChatBot/mbti-classification

loaded_model = keras.models.load_model("C:/Users/wjkim/TIKI/School/Machine/ChatBot/mbti-classification/my_saved_model")
# print(loaded_model.summary())
weights = loaded_model.get_weights()
# print(weights)


loaded_tokenizer = AutoTokenizer.from_pretrained("C:/Users/wjkim/TIKI/School/Machine/ChatBot/mbti-classification/my_saved_tokenizer")


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len

        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def predict_mbti_probabilities(sentence):
    encoded_sentence = bert_encode([sentence], loaded_tokenizer)
    # print("encoded_sen:", encoded_sentence)

    prediction = loaded_model.predict(encoded_sentence)
    print("prediction:",prediction)
    
    return dict(zip(loaded_tokenizer.classes_, prediction[0]))


# Mbti_dic = {{'ENFJ': 0.6445151, 'ENTJ': 0.0028850692, 'ESFJ': 0.005440955, 'ESFP': 0.0020330818, 'ESTJ': 0.0010679086, 'ESTP': 0.0007763918, 'ISFJ': 0.26039895, 'ISFP': 0.07188682, 'ISTJ': 0.007228388, 'ISTP': 0.0037673926}}
sentence = "I love you"
probabilities = predict_mbti_probabilities(sentence)
print(probabilities)

