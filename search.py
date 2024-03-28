#!/usr/bin/python3
import re
import nltk
import sys
import getopt
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
import math

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')
    # Initiate stemmer and vectorizer
    stemmer = PorterStemmer()

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
        # Initialise the scores board for each query
        scores_board = {}
        # Initialise a list to store the tokenized query
        query_tokens = []
        # Tokenize the query by sentence first
        query_sentence = sent_tokenize(query)
        for sentence in query_sentence:
            # Further tokenize the sentence by word
            query_tokens.extend(word_tokenize(sentence))
        # Stem the query tokens
        stemmed_query = [stemmer.stem(token.lower()) for token in query_tokens]
        # Get the posting lists of the query terms and store them in term_postings
        term_posting = get_posting(stemmed_query, dictionary, postings_file)
        # Update the scores of the documents
        score_update(term_posting, stemmed_query, total_docu_num, dictionary, scores_board)
        # Sort the scores and write the results to the output file
        sort_and_write_results(scores_board, results_file)

def sort_and_write_results(scores, results_file):
    # Sort the scores in descending order
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    counter = 0
    # Write the results to the output file
    with open(results_file, 'a') as file:
        for doc, score in sorted_scores:
            if counter < 10:
                if counter != 0:
                    file.write(' ')
                file.write(str(doc))
                counter += 1
            else:
                break
        file.write('\n')

def score_update(term_posting, stemmed_query, total_docu_num, dictionary, scores):
    for term in term_posting:
        raw_df, posting_ptr = dictionary[term]
        for doc in term_posting[term]:
            # Each document is a tuple of docID and normalized tf weight
            docID = doc[0]
            normalized_doc_tf_weight = doc[1]
            if docID not in scores:
                # Initialize the score of the document to be 0 if it is not in the dictionary
                scores[docID] = 0
            scores[docID] += get_query_term_weight(term, float(raw_df), stemmed_query, float(total_docu_num)) * normalized_doc_tf_weight
            # sum the score of each term

def get_query_term_weight(term, raw_df, stemmed_query, total_docu_num):
    # Calculate the tf-idf weight of the query term
    raw_tf = stemmed_query.count(term)
    td_weight = 1 + math.log(raw_tf, 10)
    idf_weight = math.log((total_docu_num / raw_df), 10)

    return td_weight * idf_weight

"""
Retrieve the posting lists of the tokenized query terms
"""
def get_posting(query_terms, dictionary, postings_file):
    term_posting = {}
    for term in query_terms:
        # If the term is not in the dictionary, skip it
        if term not in dictionary:
            continue
        raw_df, posting_ptr = dictionary[term]
        with open(postings_file, 'r') as file:
            posting = []
            file.seek(posting_ptr)
            line = file.readline().strip()
            pairs = line.strip().replace('(', '').replace(')', '').split(', ')
            for i in range(0, len(pairs), 2):
                doc = int(pairs[i])
                value = float(pairs[i + 1])
                posting.append((doc, value))
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
