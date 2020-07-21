import pandas as pd
import re
import string

# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = {i: 0 for i in stopwords.words('english') if len(i) > 1}
from tqdm import tqdm


def is_num(text):
    try:
        float(text)
        return True
    except:
        return False


abbrevs = {'public co ltd': 'pcl'}


def make_name_dict(name_list,
                   word_dict,
                   case=False,
                   clean=True,
                   clean_list=None,
                   abbrevs=abbrevs):
    """
  create a cleaned name list and a dictionary to match keywords against names in the name list
  
  name_list: a basic list of names(strings), or a list of lists where each embedded list contains variations of the same name
  word_dict: a dictionary with keys comprising of common english words
  case: whether or not to consider casing, defaults to False
  clean: whether or not to clean up each name by removing words in clean_list
  clean_list: list of words/phrases to remove from each name, only used if clean=True
  abbrevs: dictionary of phrase-abbreviation pairs, to convert common phrases into standard abbreviations
  """
    if clean: assert clean_list is not None
    name_list = list(name_list)
    name_dict = {}
    for i in tqdm(range(len(name_list))):
        names = name_list[i]
        if type(names) is str:
            words = names.split()
            names = [names] if case else [names.lower()]
        else:
            words = [[word for word in name.split()] for name in names]
            wrds = []
            for w in words:
                wrds.extend(w)
            words = wrds
            names = names if case else [name.lower() for name in names]

        words = [re.sub(r'\W', '', word) for word in words]

        if not case:
            words = [w.lower() for w in words]
        if clean:
            clean_list = [w.lower()
                          for w in clean_list] if case else clean_list
            # first clean out the unwanted words going into the name_dict
            words = [w for w in words if w not in clean_list]

            # clean out the names, only keep the name if it is not a commonly used single word (or anyword in the word_dict)
            clean_names = list(names)
            for name in names:
                # first convert the common phrases and abbreviations in each name
                name = name if case else name.lower()
                for phrase, abbrev in abbrevs.items():
                    name = name.replace(phrase, abbrev)
                # remove unwanted words from the name
                clean_name = ' '.join(
                    [i for i in name.split() if i not in clean_list])
                # only record the name if it is not a common word, identical to the original name, or already recorded
                if word_dict.get(
                        clean_name
                ) is None and clean_name != name and clean_name not in clean_names:
                    # also don't record numbers
                    if not is_num(clean_name):
                        clean_names.append(clean_name)
            name_list[i] = clean_names
        words = [w for w in words if w.strip() != '']
        for word in words:
            if name_dict.get(word) is None: name_dict[word] = [i]
            else: name_dict[word].append(i)

    name_dict = {word: set(inds) for word, inds in name_dict.items()}
    return name_dict, name_list


# create name_dict to check the most common words
def create_name_stuff(ners, word_dict):
    """
    create a name list and name dict from a given ners dataframe

    word_dict: dictionary of common english words
    """
    name_dict, _ = make_name_dict(ners.checker, 
                                  word_dict,
                                  case=False, clean=False)
    name_counts = {name: len(val) for name, val in name_dict.items()}
    name_counts = pd.Series(name_counts)

    # word that shows up more than 400 times should be removed
    clean_list = [i for i in name_counts.index if name_counts[i] >= 400]
    name_dict, name_list = make_name_dict(ners.checker,
                                          word_dict,
                                          case=False,
                                          clean=True,
                                          clean_list=clean_list)

    # remove names that are single word and shared amongst multiple companies
    for l in range(len(name_list)):
        names = []
        for name in name_list[l]:
            if len(name.split()) > 1:
                names.append(name)
            elif name_dict.get(name) == {l}:
                names.append(name)
            # else:
            #   print(name)
        name_list[l] = names
    return name_list, name_dict


unwanted = ['&nbsp', '']


def clean_error_decode(body):
    """
  remove decoding/encoding errors from a body of text
  """
    body = re.sub(r'\[(.*)\]', '',
                  body)  # remove anything between [] (usually names of images)
    body = re.sub(
        r'<.*?(>?)>', '',
        body)  # remove anything between <> (usually hyperlinks and html tags)

    # split into paragraph format to process each paragraph one by one
    paras = body.split('\n')
    output = ''
    for body in paras:
        # remove hyperlinks not inside <>
        # only the start of hyperlinks are used,
        # .com, .net etc. not used as might end up removing emails which shouldnt be removed here
        # need emails to identify lines/sign offs of the authors
        hyperlinks = ['http', 'www.']
        reg_hypers = [r'\.(.+)\.']
        words = body.split()
        for i in range(len(words)):
            for hl in hyperlinks:
                if hl in words[i]:
                    words[i] = ' '
            for rh in reg_hypers:
                if re.search(rh, words[i]):
                    words[i] = ' '
        body = ' '.join(words)
        body = body.replace('&nbsp', ' ')  # remove html blankspace formatting
        body = ''.join(filter(lambda x: x in string.printable, body))

        # remove wrongly decoded character
        remove = []
        for found in re.findall('=[0-9A-Z]{2}.*', body):
            find = found.split()[0][:3]
            if find not in remove:
                remove.append(find)
        for word in remove:
            body = body.replace(word, '')

        # remove byte string indicator
        if body[:2] == "b'":
            if len(body) == 2:
                return ''
            body = body[2:]

        # remove decoding errors resulting in string like \xe2\x80\x93
        while True:
            if '\\x' in body:
                ind = body.index('\\x')
                body = body[:ind] + body[ind + 4:]
            else:
                break

        output += body + '\n'

    return output


# removes line with phone numbers/email
def clean_text(body, name_dict, stops=[], case=False):
    """
  clean a body of text

  body: body of text to clean
  stops: list of strings which denote end of document's relevant portion
  name_dict: a dictionary mapping keywords to items in a name list 
  """
    body = [para for para in body.split('\n') if para]
    stop = 0
    for r in range(len(body)):
        row = body[r]
        if len(row.split()) <= 5:  # handle lines with <= 5 words
            words_ = row.split()
            words_ = words_ if case else [w_.lower() for w_ in words_]
            if all(
                    name_dict.get(word_) is None for word_ in words_
            ):  # roughly check if any part of any stock name exists in the row
                body[r] = ''
        if re.findall(r'(\(?\+?[0-9]{2,4}\)?[ \t]*[0-9]{2,4}[ \t]*[0-9]{2,4})',
                      row):  # remove phone numbers
            body[r] = ''
            continue
        if re.findall(r'@.+\.[a-z]{2,10}', row):  # remove emails
            body[r] = ''
            continue
        for st in stops:
            if st.lower() in row.lower():
                stop = r
                break
        if stop > 0: break
    if stop > 0:
        body = body[:stop]
    body = [para for para in body if para]
    return '\n'.join(body)


def remove(ele, lst):
    """
  remove all occurences of an element from a list
  """
    pops = []
    for i in range(len(lst)):
        if lst[i] == ele: pops.append(i)
    return [lst[i] for i in range(len(lst)) if i not in pops]


def find_char(char, word):
    """
  find the position of a character in a word, exactly the same thing as str.index but with error handling
  """
    assert len(char) == 1 and type(word) is str
    i = 0
    while i < len(word):
        if word[i] == char: break
        else: i += 1
    if i == len(word):
        return None
    return i


def levenshtein(word1, word2):  # O(n) variant
    """
  find the levenshtein distance between 2 words
  """
    p1, p2 = 0, 0
    dist = 0
    while p1 < len(word1) or p2 < len(word2):
        # get the current character for each word
        c1 = word1[p1] if p1 < len(word1) else None
        c2 = word2[p2] if p2 < len(word2) else None

        if c1 == c2:
            p1 += 1
            p2 += 1
        else:
            dist += 1
            # for when a word's last letter is reached
            if c1 is None: p2 += 1
            elif c2 is None:
                p1 += 1
                # other cases
            else:
                # first check if either character in question is the last of their respective word
                if p1 == len(word1) - 1 and p2 == len(word2) - 1:
                    flag1, flag2 = True, True  # if both are last characters
                elif p1 == len(word1) - 1:
                    flag1, flag2 = False, True  # if last of only word1
                elif p2 == len(word2) - 1:
                    flag1, flag2 = True, False  # if last of only word2
                    # check if current character of either word is the same as next character of other word
                else:
                    if word1[p1 + 1] == c2 and word2[p2 + 1] == c1:
                        flag1, flag2 = True, True  # ab, ba situation
                    else:
                        # consider the subword from current position onwards and find the each character
                        checker1 = find_char(c1, word2[p2:])
                        checker2 = find_char(c2, word1[p1:])
                        if checker1 is None and checker2 is None:
                            flag1, flag2 = True, True  # neither character in other word
                        elif checker1 is None:
                            flag1, flag2 = True, False  # a, bcd situation
                        elif checker2 is None:
                            flag1, flag2 = False, True  # bcd, a situation
                            # if both contain each other's current character
                        else:
                            if checker1 == checker2:
                                flag1, flag2 = True, True  # equally far away in both pairs
                            elif checker1 < checker2:
                                flag1, flag2 = False, True  # aaax, xaaa situation
                            elif checker2 < checker1:
                                flag1, flag2 = True, False  # xaaa, aaax situation
                            else:
                                print(
                                    f'word1: {word1} | {p1} | {c1} | {checker1}'
                                )
                                print(
                                    f'word2: {word2} | {p2} | {c2} | {checker2}'
                                )
                                raise Exception('unconsidered case')
                # update
                if flag1: p1 += 1
                if flag2: p2 += 1

    return dist


def path_from_scores(scores, skip=1, missing=1):
    '''
  find paths through an ordered set of positional indices called scores, searching by breadth

  scores: a list of lists containing the possible positions each point along the path can access

  skip: the total allowable difference between positions 
        e.g(if skip=0 then the allowed paths must have a difference of 1 between points such as 1,2,3,4)

  missing: allowable number of missing connections
  '''
    assert all(type(i) is list for i in scores)
    scores = [set(i) for i in scores]
    s = 1
    paths = [[i] for i in scores[0]]
    while s < len(scores):
        start_new = len(paths)
        for curr in scores[s]:
            for p in range(start_new):
                # if the end of the path is behind current in sequence
                # and if the number of positions skipped is allowable
                path = paths[p]
                if curr - path[-1] > 0 and curr - path[0] - len(path) <= skip:
                    path = path + [curr]
                    paths.append(path)
        s += 1
        paths = [p for p in paths if len(p) + missing >= s
                 ]  # keep only paths that are long enough
        # paths[start_new:] # keep only the paths that have been extended
    return paths


def find_name(text, names, stop_words, max_score=3, skip=1, skip_len=99, missed=1):
    '''
  Calculates finds a substring within a string that allows for the minimal aggregated wordwise levenshtein distance

  text: the body of text to check through

  names: the list of strings to search for/score against

  stop_words: a dictionary containing english stopwords e.g: NLTK

  max_score: highest allowable levenshtein distance for a word pair to be considered a match

  skip: allowable number of words between consecutive words in the text deemed to be part of the name

  skip_len: allowable length of words skipped

  missed: allowable number of words in the name to be considered not in the text
  '''
    assert type(names) is list
    # record the paths and clean up the text body
    outputs = {n: [] for n in names}
    text = [re.sub(r'[^A-Za-z0-9\-\%\$\.]', '', w) for w in text.split()
            ]  # keep alphanumerics and characters relating to numbers
    text = [i for i in text
            if stop_words.get(i.lower()) is None]  # remove stopwords
    for name in names:
        name_ = name  # for calling the outputs dict
        name = [re.sub(r'\W', '', n)
                for n in name.split()]  # remove non-alphanumerics

        # adjust the acceptable number of words missed based on the length of the name
        missed_ = round(len(name) / 2) - 1 if (
            len(name) / 2 <= missed) else missed
        missed_ = 0 if missed_ < 0 else missed_

        # track the positions that give the lowest score for each word in name
        scores = []
        for n in name:
            score_indices = {999: []}
            for w in range(len(text)):
                score = levenshtein(n, text[w])
                min = list(score_indices.keys())[0]
                # only record if the score is the minimum
                if score < min: score_indices = {score: [w]}
                elif score == min: score_indices[score].append(w)
            scores.append(score_indices)

        filtered_scores = [s for s in scores if list(s.keys())[0] <= max_score]
        if len(scores) - len(filtered_scores) > missed_:
            continue
        scores = [list(i.values())[0] for i in filtered_scores]

        # find the paths that exist for a set of possible indices
        paths = path_from_scores(scores, skip=skip)
        if len(paths) == 0: continue
        else: outputs[name_] = (paths)
    return outputs


def get_names(text,
              name_dict,
              stop_words,
              name_list,
              case=False,
              max_score=3,
              skip=1,
              skip_len=99,
              missed=1):
    '''
  identify the names within the name_list that are in a body of text

  text: the body of text to be searched

  name_dict: a dict mapping keywords to indices in the name_list

  name_list: list of names to identify

  stop_words: dictionary of common english stop words e.g: NLTK

  case: whether to ignore casing or not, ignores by default

  skip: how many words in the text are allowed to be skipped

  skip_len: allowable length of word to skip

  missed: how many words are allowed to be missed in a name
  '''
    words = text.split()
    matches = []
    # first find at least 1 word that matches a name
    # this creates a list of proposed names to work with
    for word in words:
        word = re.sub(r'\W', '', word)
        word = word if case else word.lower()
        if name_dict.get(word) is not None:
            matches.extend(name_dict.get(word))
    matches = set(matches)
    matches_ = list(matches)
    matches = [name_list[m] for m in matches]

    if not case:
        matches = [[m_.lower() for m_ in m] for m in matches]

    # check whether there exists some substring that sufficiently matches the name
    # keep only names that 'exist' in the text
    for m in range(len(matches)):
        match = matches[m]
        paths = find_name(text,
                          match,
                          stop_words,
                          max_score=max_score,
                          skip=skip,
                          skip_len=skip_len,
                          missed=missed)
        if all(i == [] for i in paths.values()):
            matches[m] = None
            matches_[m] = None
        else:
            matches[m] = (matches[m], paths)

    matches_ = [m for m in matches_ if m is not None]
    matches = [m for m in matches if m is not None]
    return matches, words, matches_


def best_matches(matches,
                 text,
                 word_dict,
                 n=1,
                 case=False,
                 mode='levenshtein',
                 max_score=3,
                 use_proportion=False,
                 max_proportion=0.5,
                 ):
    '''
  returns the n best from a set of matches by the mode determined

  matches: a list of matches with format [{name: paths}, ...]

  text: the text to be search, split into individual words

  word_dict: dictionary of common words to filter out common words

  n: best n results

  case: whether to ignore casing or not

  mode: levenshtein scores each match by levenshtein distance
        path scores by number of gaps in the path e.g([1,3] scores 1, [1,2] scores 0)

  max_score: highest allowable score *NOTE: this score ceiling is applied before max_proportion

  use_proportion: whether or not to compare the scores to the length of the name
                  returns the same n results but each one converted to a ratio of score:name length
  
  max_proportion: only used with use_proportion=True,
                  the maximum ratio allowable for each match
  '''
    assert mode in ['levenshtein', 'path']
    scores = {}
    min_paths = {}
    text = [i.lower() for i in text] if not case else text
    text = [re.sub(r'\W', '', i) for i in text]  # remove non-alphanumerics
    for match in matches:
        assert len(match.keys()) == 1
        name, paths = list(match.items())[0]
        clean_name = re.sub(r'[^\w\s]', '', name)
        min = 999
        min_path = []
        for path in paths:
            substring = ' '.join([text[i] for i in path])
            if mode == 'levenshtein':
                score = levenshtein(substring, clean_name)
            elif mode == 'path':
                score = path[-1] - path[0] - len(path)
            if score < min and word_dict.get(substring) is None:
                min = score
                min_path = path
        scores[name] = min
        min_paths[name] = min_path
    scores = pd.Series(scores, dtype='float').nsmallest(n, keep='all')
    scores = scores[scores <= max_score]

    if use_proportion:
        scores = scores.to_dict()
        scores = pd.Series(
            {name: score / len(name)
             for name, score in scores.items()},
            dtype='float')
        scores = scores[scores < max_proportion]
    min_paths = {
        name: min_paths[name]
        for name in min_paths if scores.get(name) is not None
    }

    scores = scores.to_dict()
    # filter out stock names with overlapping paths in the text
    positions = {i: []
                 for i in range(len(text))
                 }  # record tuples of (name, score) for each position
    for name, path in min_paths.items():
        curr_score = scores[name]
        for pos in path:
            if positions[pos] != []:
                overlap_score = positions[pos][0][1]
                # if the scores are unequal
                if curr_score != overlap_score:
                    # remove the one with the lower score from both the scores and the positions
                    if curr_score > overlap_score:
                        for p in path:
                            positions[p] = remove((name, curr_score),
                                                  positions[p])
                        scores.pop(name, None)
                    else:
                        for item in positions[pos]:
                            for p in min_paths[item[0]]:
                                positions[p] = remove(item, positions[p])
                            scores.pop(item[0], None)
                        positions[pos].append((name, curr_score))
                # if both scores are the same, => equally likely for both and thus both names should be kept
                else:
                    positions[pos].append((name, curr_score))
            # if there's nothing at the current position just record the name and score
            else:
                positions[pos].append((name, curr_score))

    return pd.Series(scores, dtype='float')

