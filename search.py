#!/usr/bin/python3
import re
import nltk
import sys
import getopt
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import math
from sklearn.feature_extraction.text import CountVectorizer

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    # This is an empty method
    # Pls implement your code in below

    # Initiate stemmer and vectorizer
    stemmer = PorterStemmer()
    vectorizer = CountVectorizer()

    # Initate the cosine similarity score dictionary and total number of documents for scoring, and term posting for accessing parts of the posting files
    scores = {} 
    term_posting = {} # query term -> posting list
    total_docu_num = 0

    # Reconstructing the dictionary from the file into memory
    dictionary = {}
    df = open(dict_file, 'r')
    dictionary_raw = df.read().split('\n')
    for line in dictionary_raw:
        if line != '':
            if line.startswith("Total_number_of_document:"): 
                total_docu_num = read_total_docu_num(line)
            else:
                term, frequency, offset = read_dictionary_line(line)
                dictionary[term] = (frequency, offset)

    # Query processing
    qf = open(queries_file, 'r')
    queries = qf.readlines()            
    for query in queries:
        query_tokens = word_tokenize(query)
        stemmed_query = [stemmer.stem(token.lower()) for token in query_tokens]
        term_posting = get_posting(stemmed_query, dictionary, postings_file) 
        score_update(term_posting, stemmed_query, total_docu_num, dictionary, scores)

def score_update(term_posting, stemmed_query, total_docu_num, dictionary, scores): # IMPORTANT: Need to account for when stemmed query word do not appear in dict
    for term in term_posting:
        raw_df, posting_ptr = dictionary[term]
        for doc in term_posting[term]:
            docID = doc[0]
            normalized_doc_tf_weight = doc[1]
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += normalized_tf_idf_weight(term, raw_df, stemmed_query, total_docu_num) * normalized_doc_tf_weight # sum the score of each term

def normalized_tf_idf_weight(term, raw_df, stemmed_query, total_docu_num): # IMPORTANT: Need to account for when stemmed query word do not appear in dict
    # Calculate the tf-idf weight of the query term
    raw_tf = stemmed_query.count(term)
    td_weight = 1 + math.log(raw_tf, 10)
    idf_weight = math.log((total_docu_num / raw_df), 10)

    # Normalize the weight


def get_posting(query_terms, dictionary, postings_file):
    term_posting = {}
    for term in query_terms:
        raw_df, posting_ptr = dictionary[term]
        with open(postings_file, 'r') as file:
            file.seek(posting_ptr)
            posting = file.readline().strip()
            term_posting[term] = posting
    return term_posting

def read_total_docu_num(line):
    total_docu_num = 0
    text, total_docu_num = line.split(' ')
    total_docu_num = total_docu_num.strip()
    return total_docu_num

def read_dictionary_line(line):
        term, frequency, offset = line.split(' ')
        term = term.strip()  
        frequency = frequency.strip() 
        offset = offset.strip()  

        frequency = int(frequency)
        offset = int(offset)

        return term, frequency, offset

dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
