#####################################################################################
#main                                                                               #
#Created by: Antonio DiMicco                                                        #
#Assignment #3                                                                      #
#Takes 5 command line arguments:                                                    #
#< stoplist file > < training data file > < training label file >                   #
#< testing data file > < testing label file >                                       #
#Run the program as such: python main.py < stoplist file > < training data file >   #
#< training label file > < testing data file > < testing label file >               #
#Works and tested with Python 2.6.6                                                 #
#####################################################################################

import sys
import math
import csv
import random
import stop_words
import goslate
import nltk
from nltk.corpus import sentiwordnet as swn
from collections import Counter

gs = goslate.Goslate()

"""
Desc: Puts data from an csv file into a 2D list
Pre:  The csv file name
Post: Returns a 2D list with csv data
"""
def import_csv(file_name):
    with open(file_name, newline='') as file:
        contents = csv.reader(file)
        csv_data = list()
        for row in contents:
            csv_data.append(row)
    return csv_data

"""
Desc: Randomly splits tweets and creates file for train_data, train_labels, test_data and test_labels
Pre:  2D list of the data
Post: 2D list containing train_data, train_labels, test_data, and test_labels
"""
def randomize_tweets(list, labels, pieces):
    size = len(list)
    start_size = size
    portion = start_size/pieces
    # random_nums = []
    data = []
    for x in range(1,pieces):
        tweet_data = []
        tweet_labels = []
        while (start_size - portion) < size:
            number = random.randint(0, size-1)
            tweet = list.pop(number)
            value = labels.pop(number)
            tweet_data.append(tweet)
            tweet_labels.append(value)
            # random_nums.append(number)
            size = size - 1
        data.append(tweet_data)
        data.append(tweet_labels)
        start_size = size
    data.append(list)
    data.append(labels)
    # data.append(random_nums)
    return data

"""
Desc: Writes the data to the next line of the file
Pre:  The name of the file and a string of data
Post: None
"""
def write_to_file(name, data):
    file = open(name, 'a+')
    file.write(data + '\n')

"""
Desc: Returns a specific column of a matrix and deletes the first entry which defines the column
Pre:  2D list and desired column
Post: 1D list of column data
"""
def get_column(matrix, i):
    column = list()
    for row in matrix:
        column.append(row[i])
    column.pop(0)
    return column

"""
Desc: Gets rid of all hashtag words and twitter user names as well as urls
Pre:  String
Post: Same string without words starting with # or @ or urls
"""
def del_twitter_words(tweet):
    tweet = tweet.lower()
    tweet = tweet.split()
    new_tweet = ""
    for word in tweet:
        if word[0] != '#' and word[0] != '@':
            if len(word) > 4 and word[0:4] == "http":
                continue
            new_tweet = new_tweet + word + " "
    return new_tweet

"""
Desc: Deletes characters from tweet
Pre:  A tweet in the form of a string and a string of the characters to delete
Post: The tweet without the given characters
"""
def del_characters(tweet, chars):
    tweet = tweet.replace(';', ' ').replace(',', ' ').replace(':', ' ').replace('.', ' ').replace("'"," ")
    tweet = tweet.translate(str.maketrans('', '', chars))
    return tweet

"""
Desc: Translates a list of italian words to english
Pre:  List of italian words
Post: 2D list with the original italian words and their translations
"""
def translations(words):
    translated_words = []
    for cur in words:
        translated = translate_word(cur)
        translated_words.append(translated)
    all_words = []
    all_words.append(words)
    all_words.append(translated_words)
    return all_words

"""
Desc: Translates a word from Italian to English
Pre:  An Italian word
Post: Corresponding English word
"""
def translate_word(word):
    print(word)
    try:
        word = gs.translate(word, 'en', 'it')
    except:
        print("herexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        word = translate_word(word)
    return word

def get_senti_values(word_list):
    word_dict = {}
    temp = 0
    for pos in range(0, len(word_list[1])):
        print(word_list[1][pos])
        values = sentiwordnet_values_average(word_list[1][pos])
        # values = sentiwordnet_values(word_list[1][pos])
        # print(values)
        word_dict[word_list[0][pos]] = values
    return word_dict

def sentiwordnet_values(word):
    value = (swn.senti_synsets(word))
    value_list = list(value)
    Positive_Average = 0
    Negative_Average = 0
    if (len(value_list) > 0):
        Positive_Average += value_list[0].pos_score()
        Negative_Average += value_list[0].neg_score()
    averages = list()
    averages.append(Positive_Average+1)
    averages.append(Negative_Average+1)
    return averages

"""
Desc: Gets the average SentiWordNet values for the given word
Pre:  An English word
Post: The SentiWordNet values for the given word or an empty list
      if the word had no values
"""
def sentiwordnet_values_average(word):
    value = (swn.senti_synsets(word))
    value_list = list(value)
    # print(value_list)
    Positive_Average = 0
    Negative_Average = 0
    count = 0
    if (len(value_list) > 0):
        for synset in value_list:
            # print(synset)
            Positive_Average += synset.pos_score()
            Negative_Average += synset.neg_score()
            count += 1
        Positive_Average = Positive_Average/float(count)
        Negative_Average = Negative_Average/float(count)
    # print(Positive_Average)
    # print(Negative_Average)
    averages = list()
    averages.append(Positive_Average+1)
    averages.append(Negative_Average+1)
    return averages

"""
Desc: Makes a list of the stop words
Pre:  The file name containing the stop words
Post: Returns a list of the stop words
"""
def stop_list(stop):
    with open(stop, 'r') as stop_string:
        stop_words = stop_string.read().replace('\n',' ')
    stop_words = stop_words.split()
    return stop_words
    
"""
Desc: Puts data from a file into a list
Pre:  The file name
Post: Returns a list of the data from the file
"""
def create_list(data):
    words = ""
    for row in data:
        words = words + ' ' + row
    words = words.split()
    return words
    
"""
Desc: Removes duplicates and stop words from a list
Pre:  The list of stop words and the list of training words
Post: Returns the vocabulary
"""
def remove_duplicates(stop, train, min_words):
    count = Counter(train)
    train_final = []
    for word in count:
        if count[word] >= min_words:
            train_final.append(word)
    train_final = list(set(train_final) - set(stop))
    train_final.sort()
    return train_final

"""
Desc: Makes a 2D list of feature vectors
Pre:  The vocabulary, labels, data, stop words, and either a 1 or 0  
      whether or not to add a column for the label
Post: Returns a 2D list with the feature vectors filled in
"""    
def set_file_data(vocab, labels, twitter_data, stop, add_row):
    width = len(vocab)
    length = len(labels)
    data = [[0 for x in range(width+add_row)] for x in range(length)]
    curr_sent = 0
    # file_data = open(file_name, 'r')
    
    # for line in file_data:
    for row in twitter_data:
        if add_row == 1:
            data[int(curr_sent)][width] = int(labels[curr_sent])
        tweet = row.split()
        tweet = list(set(tweet) - set(stop))
        for word in tweet:
            try:
                pos = vocab.index(word)
            except ValueError:
                pos = -1
            if pos != -1:
                data[int(curr_sent)][int(pos)] = 1
        curr_sent += 1
        
    return data
      
"""
Desc: Makes a list containing the number of instances of 
      each word for wise sayings, prediction sayings, and 
      total sayings, as well as the number of wise and 
      prediction sayings in general
"""  
def get_data(data, strength):
    width = len(data[0])-1
    length = len(data)
    wise_data = [(1*strength) for x in range(width)]
    predict_data = [(1*strength) for x in range(width)]
    total_data = [(2*strength) for x in range(width)]
    
    for col in range(0, width):
        for row in range(0, length):
            if data[row][col] == 1: 
                if data[row][width] == 0:
                    wise_data[col] += 1
                    total_data[col] += 1
                elif data[row][width] == 1:
                    predict_data[col] += 1
                    total_data[col] += 1

    for pos in range(0, width):
            wise_data[pos] = (wise_data[pos]/float(total_data[pos]))
            predict_data[pos] = (predict_data[pos]/float(total_data[pos]))

    wise_count = 0
    predict_count = 0
    for pos in range(0, length):
        if data[pos][width] == 0:
            wise_count += 1
        else:
            predict_count += 1
    counts = [wise_count, predict_count, length]
                    
    all_data = [wise_data, predict_data, total_data, counts]
    return all_data

def get_data_senti(data, senti_dict, strength, train_words):
    width = len(data[0])-1
    length = len(data)
    wise_and_predict_data = get_senti_priors(senti_dict, strength, train_words)
    # print(wise_and_predict_data[1])
    # print(wise_and_predict_data[0])
    # print(wise_and_predict_data[1][0])
    wise_data = wise_and_predict_data[1]
    predict_data = wise_and_predict_data[0]
    total_data = [(1*strength) for x in range(width)]

    for col in range(0, width):
        for row in range(0, length):
            if data[row][col] == 1:
                if data[row][width] == 0:
                    wise_data[col] += 1
                    total_data[col] += 1
                elif data[row][width] == 1:
                    predict_data[col] += 1
                    total_data[col] += 1

    for pos in range(0, width):
        # if wise_data[pos] >= (2 + wise_and_predict_data[1][pos]):
            wise_data[pos] = (wise_data[pos]/float(total_data[pos]))
        # else:
        #     wise_data[pos] = 1
        # if predict_data[pos] >= (2 + wise_and_predict_data[0][pos]):
            predict_data[pos] = (predict_data[pos]/float(total_data[pos]))
        # else:
        #     predict_data[pos] = 1

    wise_count = 0
    predict_count = 0
    for pos in range(0, length):
        if data[pos][width] == 0:
            wise_count += 1
        else:
            predict_count += 1
    counts = [wise_count, predict_count, length]

    all_data = [wise_data, predict_data, total_data, counts]
    return all_data

def get_senti_priors(senti_dict, strength, train_words):
    pos_prob = []
    neg_prob = []
    for pos in range(0, len(train_words)):
        pos_val = senti_dict[train_words[pos]][0]
        neg_val = senti_dict[train_words[pos]][1]
        pos_prob.append(pos_val*strength)
        neg_prob.append(neg_val*strength)
    # print(pos_prob)
    # print(neg_prob)
    return [pos_prob, neg_prob]

"""
Desc: Classifies whether a fortune is a wise saying or 
      a prediction. Uses log space for the calculations.
Pre:  The list of data from the get_data function
      as well as the feature vector for the fortune
Post: Returns 1 if it's a prediction and 0 if it's a wise saying
"""        
def classification(values, data, train_words):
    neg_prob = 0
    pos_prob = 0
    # small_value = 0.1
    for pos in range(0, len(values)):
        if values[pos] == 1:
            # print(train_words[pos])
            if data[0][pos] != 0:
                neg_prob += math.log(data[0][pos])
                # print("neg")
                # print(data[0][pos])
            # else:
            #     neg_prob += math.log(small_value)
            if data[1][pos] != 0:
                pos_prob += math.log(data[1][pos])
                # print("pos")
                # print(data[1][pos])
                # print()
            # else:
            #     pos_prob += math.log(small_value)
    neg_prob += math.log((float(data[3][0])/data[3][2]))
    pos_prob += math.log((float(data[3][1])/data[3][2]))

    # print(pos_prob)
    # print(neg_prob)
    # print()

    if neg_prob <= pos_prob:
        return 1
    else:
        return 0

"""
Desc: Makes a list of if the testing fortunes are 
      more likely to be wise sayings or predictions
Pre:  The list of data from the get_data function
      as well as the feature vectors
Post: Returns a list of the results
"""
def probability(values, data, train_words):
    # print(data[0])
    # print(data[1])
    result_labels = [0 for x in range(len(values))]
    for result in range(0,len(result_labels)):
        result_labels[result] = classification(values[result], data, train_words)
    return result_labels

"""
Desc: Checks how accurate the results are
Pre:  The list of the results and the list of the actual labels
Post: Returns the percent accuracy
"""    
def check_results(results, compare_labels):
    pos_correct = 0
    neg_correct = 0
    pos_classified = 0
    neg_classified = 0
    pos_actual = 0
    neg_actual = 0

    for pos in range(0, len(results)):
        # print(compare_labels[pos])
        if (int)(compare_labels[pos]) == 1:
            # print("here1")
            pos_actual += 1
        else:
            # print("here2")
            neg_actual += 1

        if results[pos] == 1:
            pos_classified += 1
            if results[pos] == int(compare_labels[pos]):
                pos_correct += 1
        else:
            neg_classified += 1
            if results[pos] == int(compare_labels[pos]):
                neg_correct += 1

    # print(pos_correct)
    # print(neg_correct)
    # print(pos_actual)
    # print(neg_actual)
    # print(pos_classified)
    # print(neg_classified)

    total_correct = pos_correct + neg_correct
    accuracy = float(total_correct)/len(results)
    pos_precision = float(pos_correct)/pos_classified
    neg_precision = float(neg_correct)/neg_classified
    pos_recall = float(pos_correct)/pos_actual
    neg_recall = float(neg_correct)/neg_actual

    return [accuracy, pos_precision, neg_precision, pos_recall, neg_recall]

"""
Desc: Creates and fills the file for the preprocessed data
Pre:  The list of vocabulary and the list of feature vectors
"""
def preprocessed_file(vocab, data):
    preprocessed = open('preprocessed.txt', 'w')
    for pos in range(0, len(vocab)):
        preprocessed.write(vocab[pos])
        if pos != len(vocab)-1:
            preprocessed.write(", ")
    preprocessed.write('\n')
    
    width = len(data[0])-1
    length = len(data)
    for col in range(0, width):
        for row in range(0, length):
            preprocessed.write(str(data[row][col]))
            if row != length-1:
                preprocessed.write(", ")
        preprocessed.write('\n')
    preprocessed.close()
    
"""
Desc: Creates the file with the accuracy data
Pre:  The accuracy values and the file names used
"""
def results(train_results, nb_results, senti_results, train_name, nb_name, senti_name):
    results = open('results.txt', 'w')

    results.write(train_name + '\n')
    results.write('Accuracy:           ' + train_results[0] + '\n')
    results.write('Positive Precision: ' + train_results[1] + '\n')
    results.write('Negative Precision: ' + train_results[2] + '\n')
    results.write('Positive Recall     ' + train_results[3] + '\n')
    results.write('Negative Recall     ' + train_results[4] + '\n')
    results.write('\n')

    results.write(nb_name + '\n')
    results.write('Accuracy:           ' + nb_results[0] + '\n')
    results.write('Positive Precision: ' + nb_results[1] + '\n')
    results.write('Negative Precision: ' + nb_results[2] + '\n')
    results.write('Positive Recall     ' + nb_results[3] + '\n')
    results.write('Negative Recall     ' + nb_results[4] + '\n')
    results.write('\n')

    results.write(senti_name + '\n')
    results.write('Accuracy:           ' + senti_results[0] + '\n')
    results.write('Positive Precision: ' + senti_results[1] + '\n')
    results.write('Negative Precision: ' + senti_results[2] + '\n')
    results.write('Positive Recall     ' + senti_results[3] + '\n')
    results.write('Negative Recall     ' + senti_results[4] + '\n')
    results.write('\n')

    results.close()
        
def run_naive_bayes(train_data, train_labels, test_data, test_labels, train_words, stopwords, senti_dict, strength):
    train_values = set_file_data(train_words, train_labels, train_data, stopwords, 1)
    data_values = get_data(train_values, strength)
    data_values.append(train_words)
    data_values_senti = get_data_senti(train_values, senti_dict, strength, train_words)
    data_values_senti.append(train_words)

    test_values_train = set_file_data(train_words, train_labels, train_data, stopwords, 0)
    test_results_train = probability(test_values_train, data_values, train_words)
    train_results = check_results(test_results_train, train_labels)

    print("Naive Bayes")
    test_values = set_file_data(train_words, test_labels, test_data, stopwords, 0)
    test_results_nb = probability(test_values, data_values, train_words)
    nb_results = check_results(test_results_nb, test_labels)

    print("SentiWordNet")
    test_results_senti = probability(test_values, data_values_senti, train_words)
    senti_results = check_results(test_results_senti, test_labels)

    return [train_results, nb_results, senti_results]

def senti_values_csv(min_words):
    twitter_data = import_csv("ItalianTweets.csv")
    tweet_list = get_column(twitter_data, 6)
    tweet_sentiment = get_column(twitter_data, 2)
    size = len(tweet_list)
    for tweet in range(0, size):
        tweet_list[tweet] = del_twitter_words(tweet_list[tweet])
        tweet_list[tweet] = del_characters(tweet_list[tweet], '"!@#$%^&*()_-+=1234567890?<>|[]{}\/')

    # senti_dict = {}
    stopwords = stop_words.safe_get_stop_words('it')
    senti_words = create_list(tweet_list)
    senti_words = remove_duplicates(stopwords, senti_words, min_words)
    # print(len(senti_words))
    senti_words = translations(senti_words)
    senti_list = get_senti_values(senti_words)

    with open('Senti_Values.csv', 'wt') as senti_file:
        wr = csv.writer(senti_file, lineterminator = '\n', quoting=csv.QUOTE_ALL)
        wr.writerow(['Word', 'Pos', 'Neg'])
        length = len(senti_list)
        for pos in range(0, length):
            wr.writerow(senti_list[pos])

def main(pieces, strength, min_words):
    twitter_data = import_csv("ItalianTweets.csv")
    tweet_list = get_column(twitter_data, 6)
    tweet_sentiment = get_column(twitter_data, 2)
    size = len(tweet_list)
    for tweet in range(0, size):
        tweet_list[tweet] = del_twitter_words(tweet_list[tweet])
        tweet_list[tweet] = del_characters(tweet_list[tweet], '"!@#$%^&*()_-+=1234567890?<>|[]{}\/')

    # senti_dict = {}
    stopwords = stop_words.safe_get_stop_words('it')
    senti_words = create_list(tweet_list)
    senti_words = remove_duplicates(stopwords, senti_words, min_words)
    senti_words = translations(senti_words)
    senti_dict = get_senti_values(senti_words)

    # pieces = 5
    train_and_test_data = randomize_tweets(tweet_list, tweet_sentiment, pieces)

    # total_data = [(1*strength) for x in range(width)]

    results = open('results.txt', 'w')
    results.write('Pieces = ' + str(pieces) + '\n')
    results.write('Min Words = ' + str(min_words) + '\n\n\n')
    results.close()

    for value in range(5, strength+1):
        final_train_results = [[] for x in range(5)]
        final_nb_results = [[] for x in range(5)]
        final_senti_results = [[] for x in range(5)]

        for piece in range(0, pieces):
            print(piece)
            train_data = []
            train_labels = []
            test_data = train_and_test_data[piece*2]
            test_labels = train_and_test_data[piece*2+1]
            for y in range(0,pieces):
                if piece != y:
                    train_data.extend(train_and_test_data[y*2])
                    train_labels.extend(train_and_test_data[y*2+1])
            train_words = create_list(train_data)
            train_words = remove_duplicates(stopwords, train_words, min_words)
            results = run_naive_bayes(train_data, train_labels, test_data, test_labels, train_words, stopwords, senti_dict, value)

            for x in range(0,5):
                final_train_results[x].append(results[0][x])
                final_nb_results[x].append(results[1][x])
                final_senti_results[x].append(results[2][x])

        accuracy = [0,0,0,0]
        positive_precision = [0,0,0,0]
        negative_precision = [0,0,0,0]
        positive_recall = [0,0,0,0]
        negative_recall = [0,0,0,0]

        for x in range(0,pieces):
            accuracy[0] += final_nb_results[0][x]
            accuracy[2] += final_senti_results[0][x]
            positive_precision[0] += final_nb_results[1][x]
            positive_precision[2] += final_senti_results[1][x]
            negative_precision[0] += final_nb_results[2][x]
            negative_precision[2] += final_senti_results[2][x]
            positive_recall[0] += final_nb_results[3][x]
            positive_recall[2] += final_senti_results[3][x]
            negative_recall[0] += final_nb_results[4][x]
            negative_recall[2] += final_senti_results[4][x]

        accuracy[0] /= float(pieces)
        accuracy[2] /= float(pieces)
        positive_precision[0] /= float(pieces)
        positive_precision[2] /= float(pieces)
        negative_precision[0] /= float(pieces)
        negative_precision[2] /= float(pieces)
        positive_recall[0] /= float(pieces)
        positive_recall[2] /= float(pieces)
        negative_recall[0] /= float(pieces)
        negative_recall[2] /= float(pieces)

        for y in range(0,pieces):
            accuracy[1] += (accuracy[0] - final_nb_results[0][y]) ** 2
            accuracy[3] += (accuracy[2] - final_senti_results[0][y]) ** 2
            positive_precision[1] += (positive_precision[0] - final_nb_results[1][y]) ** 2
            positive_precision[3] += (positive_precision[2] - final_senti_results[1][y]) ** 2
            negative_precision[1] += (negative_precision[0] - final_nb_results[2][y]) ** 2
            negative_precision[3] += (negative_precision[2] - final_senti_results[2][y]) ** 2
            positive_recall[1] += (positive_recall[0] - final_nb_results[3][y]) ** 2
            positive_recall[3] += (positive_recall[2] - final_senti_results[3][y]) ** 2
            negative_recall[1] += (negative_recall[0] - final_nb_results[4][y]) ** 2
            negative_recall[3] += (negative_recall[2] - final_senti_results[4][y]) ** 2

        accuracy[1] = (accuracy[1]/float(pieces)) ** 0.5
        accuracy[3] = (accuracy[3]/float(pieces)) ** 0.5
        positive_precision[1] = (positive_precision[1]/float(pieces)) ** 0.5
        positive_precision[3] = (positive_precision[3]/float(pieces)) ** 0.5
        negative_precision[1] = (negative_precision[1]/float(pieces)) ** 0.5
        negative_precision[3] = (negative_precision[3]/float(pieces)) ** 0.5
        positive_recall[1] = (positive_recall[1]/float(pieces)) ** 0.5
        positive_recall[3] = (positive_recall[3]/float(pieces)) ** 0.5
        negative_recall[1] = (negative_recall[1]/float(pieces)) ** 0.5
        negative_recall[3] = (negative_recall[3]/float(pieces)) ** 0.5

            # final_train_results[x] = float(final_train_results[x])/pieces
            # final_nb_results[x] = float(final_nb_results[x])/pieces
            # final_senti_results[x] = float(final_senti_results[x])/pieces
            # print(final_train_results[x])
            # print(final_nb_results[x])
            # print(final_senti_results[x])
            # print()

        results = open('results.txt', 'a')

        results.write('Strength = ' + str(value) + '\n\n')

        # results.write('Training Data' + '\n')
        # results.write('Accuracy:           ' + str(final_train_results[0]) + '\n')
        # results.write('Positive Precision: ' + str(final_train_results[1]) + '\n')
        # results.write('Negative Precision: ' + str(final_train_results[2]) + '\n')
        # results.write('Positive Recall     ' + str(final_train_results[3]) + '\n')
        # results.write('Negative Recall     ' + str(final_train_results[4]) + '\n')
        # results.write('\n')

        results.write('Naive Bayes' + '\n')
        results.write('Accuracy:           ' + str(accuracy[0]) + '    ' + str(accuracy[1]) + '\n')
        results.write('Positive Precision: ' + str(positive_precision[0]) + '    ' + str(positive_precision[1]) + '\n')
        results.write('Negative Precision: ' + str(negative_precision[0]) + '    ' + str(negative_precision[1]) + '\n')
        results.write('Positive Recall     ' + str(positive_recall[0]) + '    ' + str(positive_recall[1]) + '\n')
        results.write('Negative Recall     ' + str(negative_recall[0]) + '    ' + str(negative_recall[1]) + '\n')
        results.write('\n')

        results.write('SentiWordNet' + '\n')
        results.write('Accuracy:           ' + str(accuracy[2]) + '    ' + str(accuracy[3]) + '\n')
        results.write('Positive Precision: ' + str(positive_precision[2]) + '    ' + str(positive_precision[3]) + '\n')
        results.write('Negative Precision: ' + str(negative_precision[2]) + '    ' + str(negative_precision[3]) + '\n')
        results.write('Positive Recall:     ' + str(positive_recall[2]) + '    ' + str(positive_recall[3]) + '\n')
        results.write('Negative Recall:     ' + str(negative_recall[2]) + '    ' + str(negative_recall[3]) + '\n')
        results.write('\n\n')

        results.close()

    # results(final_train_results, final_nb_results, final_senti_results, 'train_data', 'Naive Bayes', 'SentiWordNet')

main(10, 5, 2)

# #nltk.download()