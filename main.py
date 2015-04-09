from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from xml.dom import minidom
from nltk import word_tokenize
from sets import Set
from nltk.corpus import wordnet as wn
import sys
from sklearn import svm, neighbors
import unicodedata
import codecs
import string
import random
import itertools


def build_sense_mapping(language):
    '''
    map all sense
    '''
    i = 0
    xmldoc = minidom.parse('data/' + language + '-train.xml')
    data = {}
    lex_list = xmldoc.getElementsByTagName('lexelt')
    for node in lex_list:
        inst_list = node.getElementsByTagName('instance')
        for inst in inst_list:
            sense_id = inst.getElementsByTagName('answer')[0].getAttribute('senseid')

            if sense_id not in data.keys():
                    
                data[sense_id] = i
                i += 1
    
    return data



def get_sense(sense_mapping, mapping_id):
    '''
    Retrieve sense id from mapping
    '''
    senseid = []

    senseid = [key for key, value in sense_mapping.items() if value == mapping_id]
    
    if len(senseid) == 1:
        return senseid[0]
    else:
        return ""


def svm_learn(train_data):
    '''
    svm learn
    '''
    classifiers = {}    

    # a classifier for each lexelt
    for lexelt in train_data.keys():

        # classifiers[lexelt] = svm.SVC(gamma=0.001, C=100.)
        classifiers[lexelt] = svm.LinearSVC()
        classifiers[lexelt].fit(train_data[lexelt]['instance_feature_data'][0:-1], train_data[lexelt]['instance_target_data'][0:-1])

    return classifiers



def neighbor_learn(train_data):
    '''
    KNeighbors learn
    '''
    classifiers = {}
    
    # a classifier for each lexelt
    for lexelt in train_data.keys():

        classifiers[lexelt] = neighbors.KNeighborsClassifier(15, weights='uniform')
        classifiers[lexelt].fit(train_data[lexelt]['instance_feature_data'], train_data[lexelt]['instance_target_data'])
    
    return classifiers


def predict_word_sense(output_file, dev_data, classifiers, sense_mapping):
    '''
    predict
    '''
    # output file
    outfile = codecs.open(output_file, encoding='utf-8', mode='w')

    # iterate through dev_data in sorted order
    for lexelt in sorted(dev_data.keys()):

        # iterate through all instances in this particular lexelt
        for i in range(len(dev_data[lexelt]['dev_instance_id'])):

            dev_instance_id = dev_data[lexelt]['dev_instance_id'][i]
            dev_instance_features = dev_data[lexelt]['dev_instance_feature_data'][i]

            # predict word sense
            prediction = classifiers[lexelt].predict(dev_instance_features)
            senseid = get_sense(sense_mapping, prediction)

            # write output
            outfile.write(replace_accented(lexelt + ' ' + dev_instance_id + ' ' + senseid + '\n'))
                
            # if random.random() > 0.99:    
            #     print dev_instance_id
            #     print dev_instance_features
            #     print dev_data[lexelt]['dev_instance_original_context'][i]
            #     print dev_data[lexelt]['context_tokens'][i]
            #     print prediction
            #     print senseid
            #     print "\n"


def process_data(context_string, language):
    '''
    Clean up all the context string 
    '''
    modified_context = context_string

    # remove special characters
    # print modified_context
    modified_context = remove_punctuations(modified_context)
    # print modified_context

    # to lower case
    modified_context = to_lower(modified_context)
    # print modified_context

    # remove numeric
    modified_context = remove_numeric(modified_context)

    # remove stop words
    modified_context = remove_stop_words(language, modified_context)

    # tokenize lexelt's context
    context_tokens = word_tokenize(modified_context)

    # stem tokens
    context_tokens = stem_context(context_tokens, language)

    return context_tokens


def extract_k_window_from_context(lString, rString, window_size, language):
    '''
    Build k window list string
    '''
    left_string = lString
    right_string = rString
    
    # remove punctuations
    left_string = remove_punctuations(left_string)
    right_string = remove_punctuations(right_string)

    # to lower case
    left_string = to_lower(left_string)
    right_string = to_lower(right_string)

    # remove numeric
    left_string = remove_numeric(left_string)
    right_string = remove_numeric(right_string)

    # remove stop words
    left_string = remove_stop_words(language, left_string)
    right_string = remove_stop_words(language, right_string)

    # tokenize
    k_words_before_target = word_tokenize(left_string)
    k_words_after_target = word_tokenize(right_string)

    # stem tokens
    k_words_before_target = stem_context(k_words_before_target, language)
    k_words_after_target = stem_context(k_words_after_target, language)

    return k_words_before_target[-window_size:] + k_words_after_target[:window_size]



def stem_context(context_tokens, language):

    if language in ['English', 'Spanish']:
        sBall_stemmer = SnowballStemmer(language.lower())
        stemmed_context_tokens = [sBall_stemmer.stem(token) for token in context_tokens]

        return stemmed_context_tokens

    else:
        return context_tokens


def remove_stop_words(language, input_string):
    '''
    Remove stop words from inputted context
    '''

    if language == 'Catalan':
        return input_string

    stop_words_list = stopwords.words(language.lower())

    # convert input string to a list 
    # remove any stop words
    # convert back to string
    content_list = input_string.split(" ")
    modified_string = " ".join(w for w in content_list if w not in stop_words_list)
    
    # if random.random() > 0.99:
    #     print content_list
    #     print modified_string
    #     print 

    return modified_string


def remove_punctuations(input_string):
    '''
    Remove punctuations
    '''
    translate_to=u''
    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char), translate_to) for char in not_letters_or_digits)
    return input_string.translate(translate_table)


def to_lower(input_string):
    '''
    Change string to lowercase
    '''
    return input_string.lower()



def remove_numeric(input_string):
    '''
    Remove numeric from context string
    '''
    return ''.join(i for i in input_string if not i.isdigit())




def get_target_word_hyponyms(target_word, pos, language):
    '''
    Get list of hyponyms of different senses for target words
    '''
    synset = wn.synsets(target_word, pos=pos, lang=language)

    hyponyms_list = []
    
    for ss in synset:
        if len(ss.hyponyms()) > 0:
            # takes the first lemma_names from the first 5 hyponyms
            tmp = [name.lemma_names()[0] for name in ss.hyponyms()[:5]]
            hyponyms_list.append(tmp)

    hyponyms = [ls for ls in hyponyms_list]

    if len(hyponyms) > 1:
        return hyponyms[0] + hyponyms[1]
    elif len(hyponyms) > 0:
        return hyponyms[0]
    else:
        return []


    # if random.random() > 0.99:

    #     print target_word, pos, language

        # for ss in synset:
        #     if len(ss.hyponyms()) > 0:
        #         # takes the first lemma_names from the first 5 hyponyms
        #         tmp = [name.lemma_names()[0] for name in ss.hyponyms()[:5]]
        #         hyponyms_list.append(tmp)

        # print list(itertools.chain.from_iterable(hyponyms_list))
        # print hyponyms_list
        # print [ls[0] for ls in hyponyms_list]



def get_target_word_hypernyms(target_word, pos, language):
    '''
    Get a list of hypernyms of different senses for target word
    '''
    synset = wn.synsets(target_word, pos=pos, lang=language)

    hypernyms_list = []

    for ss in synset:
        if len(ss.hypernyms()) > 0:
            tmp = [name.lemma_names()[0] for name in ss.hypernyms()[:5]]
            hypernyms_list.append(tmp)

    hypernyms = [ls for ls in hypernyms_list]

    if len(hypernyms) > 1:
        return hypernyms[0] + hypernyms[1]
    elif len(hypernyms) > 0:
        return hypernyms[0]
    else:
        return []


def parse_dev_data(dev_file, train_data, language):
    '''
    Retrieve dev data
    '''

    # set parsing configuration
    if language == 'English':
        context = 'context'
    else:
        context = 'target'


    # file specific ds
    data = {}
    xmldoc = minidom.parse(dev_file)
    lexelt_list = xmldoc.getElementsByTagName('lexelt')

    for lex in lexelt_list:

        lexelt = lex.getAttribute('item')

        # lexelt specific ds
        data[lexelt] = {}

        # lexelt specific data
        data[lexelt]['dev_instance_feature_data'] = []
        data[lexelt]['dev_instance_id'] = []
        data[lexelt]['dev_instance_original_context'] = []
        data[lexelt]['context_tokens'] = []
        
        # get k_window_union_set specific for this lexelt
        lexelt_k_word_window_set = train_data[lexelt]['lexelt_k_word_window_set']

        inst_list = lex.getElementsByTagName('instance')

        # iterate through lexelt individual instances
        for inst in inst_list:

            instance_id = inst.getAttribute('id')
            l = inst.getElementsByTagName(context)[0]
            original_context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')


            # process data
            context_tokens = process_data(original_context, language)


            # a temp list that store wc
            dev_instance_word_count_list = []

            # build word_count list for every words in instance's context against union_set
            for word in lexelt_k_word_window_set:
                if word in context_tokens:
                    dev_instance_word_count_list.append(context_tokens.count(word))
                else:
                    dev_instance_word_count_list.append(0)

            # append dev_instance_features data (word count) for this instance
            data[lexelt]['dev_instance_id'].append(instance_id)
            data[lexelt]['dev_instance_feature_data'].append(dev_instance_word_count_list)
            data[lexelt]['dev_instance_original_context'].append(original_context)
            data[lexelt]['context_tokens'].append(context_tokens)

    return data


def parse_train_data(train_file, window_size, sense_mapping, language):
    '''
    Retrieve train data
    '''

    # file specific dict
    data = {}

    # language map
    lang = {
        'English' : 'en',
        'Spanish' : 'spa',
        'Catalan' : 'cat'
    }
    
    xmldoc = minidom.parse(train_file)
    lexelt_list = xmldoc.getElementsByTagName('lexelt')

    for lex in lexelt_list:

        lexelt = lex.getAttribute('item')

        # get hypernyms & hyponyms of lexelt
        wordnet_features = []
        if (language == 'English'):

            context = 'context'

            lexelt_info = lexelt.split('.')
            hyponyms = get_target_word_hyponyms(lexelt_info[0], lexelt_info[1], lang[language])
            hypernyms = get_target_word_hypernyms(lexelt_info[0], lexelt_info[1], lang[language])
            hyponyms = stem_context(hyponyms, language)
            hypernyms = stem_context(hypernyms, language)
            wordnet_features = hyponyms + hypernyms # a list

        else:
            context = 'target'


        # lexelt specific dict
        data[lexelt] = {}

        # lexelt specific data
        data[lexelt]['instance_feature_data'] = []
        data[lexelt]['instance_target_data'] = []
        data[lexelt]['instance_id'] = []
        data[lexelt]['instance_original_context'] = []
        data[lexelt]['lexelt_k_word_window_set'] = {}

        # lexelt specific word window dict field
        lexelt_k_window_union_set = Set() 

        # add in the wordnet_features into the k_window set
        lexelt_k_window_union_set = lexelt_k_window_union_set.union(wordnet_features)

        inst_list = lex.getElementsByTagName('instance')

        # iterate through lexelt individual instances
        # populate k_window_union_set
        for inst in inst_list:

            instance_id = inst.getAttribute('id')
            l = inst.getElementsByTagName(context)[0]
            original_context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')
            k_window_context = extract_k_window_from_context(l.childNodes[0].nodeValue, l.childNodes[2].nodeValue, window_size, language)
            sense_element = inst.getElementsByTagName('answer')[0]
            target_sense_id = sense_element.getAttribute('senseid')

            # ignore senseid == "U"
            if target_sense_id == 'U':
                continue

            # build the k_word_window_set
            lexelt_k_window_union_set = lexelt_k_window_union_set.union(k_window_context)

            data[lexelt]['instance_id'].append(instance_id)
            data[lexelt]['instance_target_data'].append(sense_mapping[target_sense_id])
            data[lexelt]['instance_original_context'].append(original_context)

        # iterate through lexalt individual instances
        # get k_window_union_set word count
        for inst in inst_list:

            l = inst.getElementsByTagName(context)[0]
            original_context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')
            sense_element = inst.getElementsByTagName('answer')[0]
            target_sense_id = sense_element.getAttribute('senseid')

            # ignore senseid == "U"
            if target_sense_id == 'U':
                continue

            # process data
            context_tokens = process_data(original_context, language)


            # position of the word count list dictate by the lexalt_k_window_union_set
            instance_word_count_list = []

            # word count for this every context words in this instance
            for word in lexelt_k_window_union_set:
                if word in context_tokens:
                    # append number of word
                    instance_word_count_list.append(context_tokens.count(word))
                else:
                    # word not exist in lexalt_k_window
                    instance_word_count_list.append(0)

            # append instance_feature_data (word count) for this instance
            data[lexelt]['instance_feature_data'].append(instance_word_count_list)

        # at the end of the lexelt. Then add info (for use in dev)
        # unique for each lexelt
        data[lexelt]['lexelt_k_word_window_set'] = lexelt_k_window_union_set

    return data



def replace_accented(input_str):
    '''
    Replace accent of foreign language
    '''
    nkfd_form = unicodedata.normalize('NFKD', input_str)

    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])



if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print 'Usage: python main.py <input_file> <output_file> <language>'
        sys.exit(0)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    language = sys.argv[3]

    window_size = 10
    sense_mapping = build_sense_mapping(language)

    train_data = parse_train_data(input_file, window_size, sense_mapping, language)

    svm_classifiers = svm_learn(train_data)
    nei_classifiers = neighbor_learn(train_data)

    dev_data = parse_dev_data("data/" + language + '-dev.xml', train_data, language)

    predict_word_sense(output_file, dev_data, svm_classifiers, sense_mapping)
