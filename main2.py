from nltk.corpus import stopwords
from nltk.stem.porter import *
from xml.dom import minidom
from nltk import word_tokenize
from sets import Set
import sys
from sklearn import svm, neighbors
import unicodedata
import codecs

def build_sense_mapping(language='English'):
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
    for key, value in sense_mapping.items():
        if value == mapping_id:
            return key



def extract_k_window_from_context(lString, rString, window_size):
    '''
    Build k window list string
    '''
    k_words_before_target = word_tokenize(lString)[-window_size:]
    k_words_after_target = word_tokenize(rString)[:window_size]
    
    return k_words_before_target + k_words_after_target



def svm_learn(train_data):
    '''
    svm learn
    '''
    classifiers = {}    

    for lexelt in train_data.keys():

        classifiers[lexelt] = svm.SVC()
        classifiers[lexelt].fit(train_data[lexelt]['instance_feature_data'][0:-1], train_data[lexelt]['instance_target_data'][0:-1])

    return classifiers



def neighbor_learn(train_data):
    '''
    KNeighbors learn
    '''
    classifiers = {}
    
    for lexelt in train_data.keys():

        classifiers[lexelt] = neighbors.KNeighborsClassifier()
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
            # outfile.write(lexelt + ' ' + dev_instance_id + ' ' + senseid + '\n')
            print dev_instance_id
            print dev_instance_features
            print prediction
            print senseid
            print "\n"

            outfile.write(lexelt + ' ' + dev_instance_id + ' ' + senseid + '\n')


def parse_dev_data(dev_file, train_data):
    '''
    Retrieve dev data
    '''

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
        
        # get k_window_union_set specific for this lexelt
        lexelt_k_word_window_set = train_data[lexelt]['lexelt_k_word_window_set']

        inst_list = lex.getElementsByTagName('instance')

        # iterate through lexelt individual instances
        for inst in inst_list:

            instance_id = inst.getAttribute('id')
            l = inst.getElementsByTagName('context')[0]
            original_context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')

            # tokenize lexelt's context
            context_tokens = word_tokenize(original_context)

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

    return data


def parse_train_data(train_file, window_size, sense_mapping):
    '''
    Retrieve train data
    '''

    # file specific dict
    data = {}
    
    xmldoc = minidom.parse(train_file)
    lexelt_list = xmldoc.getElementsByTagName('lexelt')

    for lex in lexelt_list:

        lexelt = lex.getAttribute('item')

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

        inst_list = lex.getElementsByTagName('instance')

        # iterate through lexelt individual instances
        # populate k_window_union_set
        for inst in inst_list:

            instance_id = inst.getAttribute('id')
            l = inst.getElementsByTagName('context')[0]
            original_context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')
            k_window_context = extract_k_window_from_context(l.childNodes[0].nodeValue, l.childNodes[2].nodeValue, window_size)
            sense_element = inst.getElementsByTagName('answer')[0]
            target_sense_id = sense_element.getAttribute('senseid')

            # build the k_word_window_set
            lexelt_k_window_union_set = lexelt_k_window_union_set.union(k_window_context)

            data[lexelt]['instance_id'].append(instance_id)
            data[lexelt]['instance_target_data'].append(sense_mapping[target_sense_id])
            data[lexelt]['instance_original_context'].append(original_context)

        # iterate through lexalt individual instances
        # get k_window_union_set word count
        for inst in inst_list:

            l = inst.getElementsByTagName('context')[0]
            original_context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')

            # tokenize lexalt's context
            context_tokens = word_tokenize(original_context)

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


if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print 'Usage: python main.py <input_file> <output_file> <language>'
        sys.exit(0)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    language = sys.argv[3]

    window_size = 10
    sense_mapping = build_sense_mapping()

    train_data = parse_train_data("data/" + language + "-train.xml", window_size, sense_mapping)

    svm_classifiers = svm_learn(train_data)
    nei_classifiers = neighbor_learn(train_data)

    dev_data = parse_dev_data("data/" + language + '-dev.xml', train_data)

    predict_word_sense("English.baseline", dev_data, nei_classifiers, sense_mapping)    

