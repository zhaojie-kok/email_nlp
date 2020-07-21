import torch
# from transformers import BertForSequenceClassification, BertTokenizer

import numpy as np
import pandas as pd
from tqdm import tqdm
import NER_utils


# finbert = BertForSequenceClassification.from_pretrained('finbert_sentiment',
#                                                         num_labels=3)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_predictions(finbert, tokenizer, sent, device):
    """
  simple function for retrieving predictions from finbert model from PROSUS AI

  finbert: finbert or some variant otherwise

  tokenizer: suitable tokenizer for finbert

  sent: sentence to process

  device: torch.device object
  """
    # tokenize the sentence and send to GPU
    toks = tokenizer.tokenize(sent)
    toks = ['[CLS]'] + toks[:510] + ['[SEP]']
    ids = tokenizer.convert_tokens_to_ids(toks)
    ids = torch.tensor([ids])
    ids = ids.to(device)

    # make prediction then remove sentence tensor from GPU
    finbert.eval()
    pred = finbert(ids)
    # del ids

    # normalise the output between 0 and 1 (finbert doesnt normalise the input)
    # pred = torch.nn.Sigmoid()(pred)
    pred = torch.nn.Softmax(dim=1)(pred).flatten()
    predval = torch.argmax(pred).item()

    return pred, predval


# Convert sentiment to numbers
pred_ids = {0: 1, 1: -1, 2: 0}


def predict(email_list,
            name_dict,
            name_list,
            stop_words,
            word_dict,
            finbert,
            tokenizer,
            device,
            pred_ids=pred_ids):
    """
  wrapper function for obtaining sentiments and entities on a per paragraph basis on texts within an email_list

  email_list: email_list generated using imap_for_gmail_hyperlinks notebook
  name_dict: dictionary mapping keywords to items in a name_list
  name_list: list containing lists of variations of each name
  stop_words: dictionary of stopwords e.g: NLTK
  word_dict: dictionary of common english words
  finbert: pytorch model with outputs mapped to pred_ids
  tokenizer: tokenizer for finbert
  device: torch.device object
  pred_ids: mapping to convert finbert outputs to sensible format
  """
    for i in tqdm(range(len(email_list))):
        email = email_list[i]
        for j in range(len(email)):
            texts = [
                text for text in email[j]['payload'] if text.strip() != ''
            ]
            texts += [[text for text in hl.values() if text.strip() != '']
                      for hl in email[j]['hyperlinks']]
            email_list[i][j]['texts'] = texts
            all_ents = {}

            for text in texts:
                if not type(text) is str:
                    continue  # ignore anything that isnt a string
                text = NER_utils.clean_error_decode(text)
                text = NER_utils.clean_text(text, name_dict=name_dict)
                text = text.split('\n')
                curr_ents = {}
                prev_ents = {}
                curr_para_ents = {}
                for para in text:
                    # reset the entity tracking for each paragraph
                    prev_para_ents = curr_para_ents  # only 1 paragraph prior is tracked
                    curr_para_ents = {}
                    sentences = para.split(
                        '. '
                    )  # use fullstop and spacing to split into sentences
                    curr_ents = {}
                    # prev_para_flag = False
                    for sent in sentences:
                        prev_ents = curr_ents
                        curr_ents = {}

                        # find all the possible stocks and record
                        ms = NER_utils.get_names(sent, name_dict, stop_words, name_list)
                        ms_ = []
                        ms_index = []
                        for idx, k in enumerate(ms[0]):
                            for name, paths in k[1].items():
                                if len(paths) > 0:
                                    ms_.append({name: paths})
                                    ms_index.append(ms[2][idx])
                        bm = NER_utils.best_matches(ms_,
                                          ms[1],
                                          word_dict,
                                          n=10,
                                          max_score=5,
                                          use_proportion=True,
                                          max_proportion=0.3)

                        stock_name = None
                        for stock_name_ in bm.keys():
                            for msi in ms_index:
                                if stock_name_ in name_list[msi]:
                                    stock_name = name_list[msi][0]
                            if not stock_name is None:
                                if all_ents.get(stock_name) is None:
                                    all_ents[stock_name] = 0
                                if curr_para_ents.get(stock_name) is None:
                                    curr_para_ents[stock_name] = 0
                                curr_ents[stock_name] = 0

                        # use previous entities if current sentence has none
                        if len(curr_ents.keys()) < 1:
                            if len(prev_ents.keys()) > 0:
                                curr_ents = {
                                    key: 0
                                    for key in prev_ents.keys()
                                }
                            else:
                                curr_ents = {
                                    key: 0
                                    for key in prev_para_ents.keys()
                                }
                                # prev_para_flag = True

                        # get sentiment predictions from finbert
                        pred, predval = get_predictions(
                            finbert, tokenizer, sent, device)

                        # only record the sentiment if the confidence (by taking softmax) is greater than 0.5
                        if pred[predval].item() > 0.5:
                            predval = pred_ids[predval]
                            for key in curr_ents.keys():
                                curr_ents[key] += predval / len(
                                    curr_ents.keys())
                                all_ents[key] += predval / len(
                                    curr_ents.keys())

                # record the entities and their scentiment score
                email_list[i][j]['entities'] = all_ents
    return email_list
