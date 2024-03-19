# For macOS, sometimes a hidden .DS_Store file maybe created in reuters file that needs to be deleted

#!/usr/bin/python3
import os
import math
import nltk
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import sys
import getopt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('Indexing...')

    # Initiate three dictionaries for term-df, term-postings, and docID-length
    doc_freq = {}
    postings_lists = {}
    length = {}

    # Initiate stemmer and vectorizer
    stemmer = PorterStemmer()
    vectorizer = CountVectorizer()

    filenames = sorted(os.listdir(in_dir), key=int)
    for filename in filenames:
        # print(filename)
        with open(os.path.join(in_dir, filename), 'r') as file:
            words = word_tokenize(file.read())
            stemmed_words = [stemmer.stem(word.lower()) for word in words] # a list of stemmed words for counting raw document frequency later

            # Compute the length of the document for cosine normalization
            doc_term_matrix = vectorizer.fit_transform(stemmed_words)
            doc_term_array = doc_term_matrix.toarray()
            term_frequencies = np.sum(doc_term_array, axis=0)
            doc_length = np.linalg.norm(term_frequencies)
            length[filename] = doc_length

            # Update dictionary and posting list for each term encountered
            terms = set(stemmed_words)
            for term in terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1
                raw_tf = stemmed_words.count(term)
                if term not in postings_lists:
                    postings_lists[term] = []    
                postings_lists[term].append((int(filename), raw_tf)) # store the docID + raw tf in postings list
    
    print("Normalizing documents...")
    # Convert the raw tf for all documents of each term into normalized log 10 based tf
    for term in doc_freq:
        normalized_postings_list = []
        for id_tf_pair in postings_lists[term]:
            # print("This is the id_tf_pair")
            # print(id_tf_pair)
            doc_id = str(id_tf_pair[0])
            raw_tf = id_tf_pair[1]
            # print("This is raw tf")
            # print(raw_tf)
            tf_weight = math.log(raw_tf,10) + 1
            # print("This is log base 10 tf")
            # print(tf_weight)
            normalized_tf_weight = tf_weight/length[doc_id]
            # print("This is normalized tf")
            # print(normalized_tf_weight)
            id_tf_pair = (doc_id, normalized_tf_weight)
            normalized_postings_list.append(id_tf_pair)
        postings_lists[term] = normalized_postings_list
    
    print("Writing...")
    # Write total document number, final dictionary and posting list to respective files
    with open(out_dict, 'a') as dict_file, open(out_postings, 'a') as postings_file:
        dict_file.write(f"Total_number_of_document: {len(length)}\n")
        for term in sorted(doc_freq):
            pointer = postings_file.tell()
            postings_file.write(f"{postings_lists[term]}\n")
            dict_file.write(f"{term} {doc_freq[term]} {pointer}\n")

input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
