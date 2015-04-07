from nltk.corpus import stopwords
from nltk.stem.porter import *
from xml.dom import minidom
from nltk import word_tokenize
from sets import Set
import sys
from sklearn import svm, neighbors
import unicodedata
import codecs


def parse_train_data(language, k, sense_mapping):

    language_lower_case = language.lower()
    training_file = ("data/" + language + '-train.xml')
    xmldoc = minidom.parse(training_file)
    data = {}
    lex_list = xmldoc.getElementsByTagName('lexelt')
    num_lexelt = 0

    # initialize a stemmer to be used later
    stemmer = PorterStemmer()

    # get language stopwords
    stop_words = stopwords.words(language_lower_case)

    # parse and process the train data
    for lex_node in lex_list:

        num_lexelt_instance = 0
        num_lexelt += 1        

        lexelt = lex_node.getAttribute('item')
        tmp_set = Set()
        data[lexelt] = {}
        data[lexelt]['data'] = []
        data[lexelt]['target'] = []
        data[lexelt]['instance_id'] = []
        data[lexelt]['original_context'] = []
        data[lexelt]['tmp_list'] = []
        data[lexelt]['features_set'] = {}

        inst_list = lex_node.getElementsByTagName('instance')
        
        for inst in inst_list:
           
            instance_id = inst.getAttribute('id')
            
            num_lexelt_instance += 1

            l = inst.getElementsByTagName('context')[0] # use only the first context
            original_context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '') # parse context
            k_window_context = extract_k_window_from_context(l.childNodes[0].nodeValue, l.childNodes[2].nodeValue, k)
            # k_window_context_no_stop_words = (w for w in k_window_context if w.lower() not in stop_words)

            answer_ele = inst.getElementsByTagName('answer')[0] # get the first answer for this instance
            answer_sense_id = answer_ele.getAttribute('senseid')

            # get only root of word
            # stems = [stemmer.stem(w) for w in k_window_context_no_stop_words]
            tmp_set = tmp_set.union(k_window_context)
            

            data[lexelt]['target'].append(sense_mapping[answer_sense_id])
            data[lexelt]['instance_id'].append(instance_id)
            data[lexelt]['original_context'].append(original_context)

        # for each instance in lexelt ~ a dict/vector of how many times each of the words in S occurred
        for inst in inst_list:

            l = inst.getElementsByTagName('context')[0]
            original_context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')
            
            # context_no_stop_words = (w for w in original_context if w.lower() not in stop_words)
            # original_context_tokens = word_tokenize(original_context)
            # context_tokens = [w for w in original_context_tokens if w not in stop_words]
            
            # print context_tokens

            # context_tokens = [stemmer.stem(w) for w in context_no_stop_words]
            context_tokens = word_tokenize(original_context)
            
            tmp_dict = {}
            tmp_list = []

            for word in tmp_set:

                if word in context_tokens:
                    
                    tmp_dict[word] = context_tokens.count(word)
                    tmp_list.append(tmp_dict[word])

                else:
                    tmp_dict[word] = 0
                    tmp_list.append(0)                    

            data[lexelt]['data'].append(tmp_dict)
            data[lexelt]['tmp_list'].append(tmp_list)
            
        data[lexelt]['features_set'] = tmp_set


        #print num_lexelt_instance, lexelt

    #print 'number_of_lexelt', num_lexelt

    #print len(data['activate.v']['data'][0].values())
    # print data['activate.v']['tmp_list'][0]
    #print data['activate.v']['target'][0:20]
    # print get_sense(sense_mapping, data['activate.v']['target'][0])
    #print data['activate.v']['instance_id'][0]
    # print data['activate.v']['context'][0]
    #print len(data['activate.v']['data'][0])

    return data



def get_dev_features(language, train_data):

    language_lower_case = language.lower()
    dev_file = ("data/" + language + "-dev.xml")
    xmldoc = minidom.parse(dev_file)
    data = {}
    lex_list = xmldoc.getElementsByTagName('lexelt')

    # initialize a stemmer to be used later
    stemmer = PorterStemmer()

    # get language stopwords
    stop_words = stopwords.words(language_lower_case)

    for lex_node in lex_list:
                
        lexelt = lex_node.getAttribute('item')
        data[lexelt] = {}
        data[lexelt]['data'] = []
        data[lexelt]['instance_id'] = []
        data[lexelt]['context'] = []
        data[lexelt]['tmp_dict'] = []
        data[lexelt]['original_context'] = []

        # get training features set
        features_set = train_data[lexelt]['features_set']

        inst_list = lex_node.getElementsByTagName('instance')

        for inst in inst_list:

            instance_id = inst.getAttribute('id')        

            l = inst.getElementsByTagName('context')[0]
            original_context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')

            # remove stop words
            # context_no_stop_words = [w for w in original_context if w.lower() not in stop_words]
            # original_context_tokens = word_tokenize(original_context)

            # context_tokens = [w for w in original_context_tokens if w.lower() not in stop_words]

            # get word stem
            # context_tokens = [stemmer.stem(w) for w in context_no_stop_words]
            context_tokens = word_tokenize(original_context)

            temp_dict = {} 
            tmp_list = []

            for word in features_set:

                if word in context_tokens:

                    temp_dict[word] = context_tokens.count(word)
                    tmp_list.append(context_tokens.count(word))

                else:
                    temp_dict[word] = 0
                    tmp_list.append(0)

            data[lexelt]['tmp_dict'].append(temp_dict)
            data[lexelt]['data'].append(tmp_list)
            data[lexelt]['instance_id'].append(instance_id) 
            data[lexelt]['original_context'].append(original_context)

    return data



def svm_learn(train_data):

    classifiers = {}    

    for lexelt in train_data.keys():

        classifiers[lexelt] = svm.SVC()
        classifiers[lexelt].fit(train_data[lexelt]['tmp_list'][0:-1], train_data[lexelt]['target'][0:-1])

    return classifiers


def neighbor_learn(train_data):

    classifiers = {}
    
    for lexelt in train_data.keys():

        classifiers[lexelt] = neighbors.KNeighborsClassifier()
        classifiers[lexelt].fit(train_data[lexelt]['tmp_list'], train_data[lexelt]['target'])
    
    return classifiers



def map_sense(language='English'):
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


    for key, value in sense_mapping.items():
        if value == mapping_id:
            return key



def extract_k_window_from_context(lString, rString, window_size):

    k_words_before_target = word_tokenize(lString)[-window_size:]
    k_words_after_target = word_tokenize(rString)[:window_size]
    
    return k_words_before_target + k_words_after_target




def replace_accented(input_str):

    nkfd_form = unicodedata.normalize('NFKD', input_str)

    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])



def predict_word_sense(language, dev_data, classifiers):

    outfile = codecs.open(language + '.baseline', encoding = 'utf-8', mode = 'w')
    
    sense_mapping = map_sense()

    for lexelt in sorted(dev_data.keys()):

        tmp_dict = dev_data[lexelt]['tmp_dict']
        tmp_data_list = dev_data[lexelt]['data']
        tmp_inst_id_list = dev_data[lexelt]['instance_id']
        tmp_context_list = dev_data[lexelt]['original_context']

        for i in range(len(tmp_data_list)):

            dev_tmp_dict = tmp_dict[i]
            dev_feature_data = tmp_data_list[i]
            dev_instance_id = tmp_inst_id_list[i]
            dev_tmp_context = tmp_context_list[i]
            prediction = classifiers[lexelt].predict(dev_feature_data)
            sense_id = get_sense(sense_mapping, prediction)
            outfile.write(lexelt + " " + dev_instance_id + " " + sense_id + "\n")
            
    outfile.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: python main.py <input_file> <output_file> <language>'
        sys.exit(0)


    language = sys.argv[3]
    sense_mapping = map_sense()
    train_data = parse_train_data(language, 10, sense_mapping)

    dev_data = get_dev_features(language, train_data)

    svm_classifiers = svm_learn(train_data)
    nei_classifiers = neighbor_learn(train_data)

    for i in range(len(dev_data['operate.v']['data'])):
        
        nei_prediction5 = nei_classifiers['operate.v'].predict(dev_data['operate.v']['data'][i])
        print nei_prediction5, "\n", get_sense(sense_mapping, nei_prediction5), "\n", dev_data['operate.v']['instance_id'][i], "\n" ,dev_data['operate.v']['data'][i], "\n\n", dev_data['operate.v']['original_context'][i], "\n\n", dev_data['operate.v']['tmp_dict'][i],"\n\n\n"

        print "\n\n"

    for i in range(len(dev_data['activate.v']['data'])):

        nei_prediction6 = nei_classifiers['activate.v'].predict(dev_data['activate.v']['data'][i])
        print nei_prediction6, "\n", get_sense(sense_mapping, nei_prediction6), "\n", dev_data['activate.v']['instance_id'][i], "\n", dev_data['activate.v']['data'][i], "\n\n", dev_data['activate.v']['original_context'][i], "\n\n", dev_data['activate.v']['tmp_dict'][i], "\n\n\n"

        print "\n\n"

    # predict_word_sense(language, dev_data, nei_classifiers)




