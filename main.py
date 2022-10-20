import json

import nltk
import requests


def main(url):
    data = requests.get(url)
    data = data.text

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    tokens = nltk.tokenize.word_tokenize(data)
    tagged_words = nltk.pos_tag(tokens)
    tags = [word_tag[1] for word_tag in tagged_words]

    part_of_speech_dict = {
        'Имя существительное': ['NN', 'NNP', 'NNPS', 'NNS'],
        'Прилагательное': ['JJ', 'JJR', 'JJS'],
        'Глаголы': ['VB', 'VBP', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'Наречия': ['RBR', 'RBS'],
        'Междометия': ['IN'],
        'Предлоги': ['PRP', 'PRPS'],
    }

    part_of_speech_list = []

    for name in part_of_speech_dict:
        tags_from_group = list(
            filter(lambda x: x in part_of_speech_dict[name], tags))
        part_of_speech_list.append([len(tags_from_group), name])

    part_of_speech_list.sort(reverse=True)

    result_dict = {}

    # В результате JSON будет содержать топ-5 частей речи
    for number_and_name in part_of_speech_list[:5]:
        result_dict[number_and_name[1]] = number_and_name[0]

    result_json = json.dumps(result_dict, indent=4, ensure_ascii=False,)

    return result_json


if __name__ == '__main__':
    url = 'https://gist.githubusercontent.com/nzhukov/b66c831ea88b4e5c4a044c' \
        '952fb3e1ae/raw/7935e52297e2e85933e41d1fd16ed529f1e689f5/A%2520Brief' \
        '%2520History%2520of%2520the%2520Web.txt'
    print(main(url))
