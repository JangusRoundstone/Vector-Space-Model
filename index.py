# For macOS, sometimes a hidden .DS_Store file maybe created in reuters file that needs to be deleted

#!/usr/bin/python3
import os
import math
import nltk
from nltk.corpus import reuters
from nltk.tokenize import sent_tokenize
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

    # Read each document from the directory
    filenames = sorted(os.listdir(in_dir), key=int)
    for filename in filenames:
        with open(os.path.join(in_dir, filename), 'r') as file:
            words = []
            sentences = sent_tokenize(file.read())
            for sentence in sentences:
                words.extend(word_tokenize(sentence))
            stemmed_words = [stemmer.stem(word.lower()) for word in
                             words]  # a list of stemmed words for counting raw document frequency later

            log_term_frequencies = {}
            # check_count = 0

            # Count the raw document frequency for each term and update the dictionary (doc_freq)
            terms = set(stemmed_words)
            for term in terms:
                doc_freq[term] = doc_freq.get(term, 0) + 1
                raw_tf = stemmed_words.count(term)

                # Convert the raw tf for all documents of each term into normalized log 10 based tf
                tf = math.log(raw_tf, 10) + 1
                # Store the log tf for each term in the dictionary (log_term_frequencies{})
                log_term_frequencies[term] = tf

            # Compute the length of the document for cosine normalization
            doc_length = np.linalg.norm(list(log_term_frequencies.values()))
            length[filename] = doc_length

            # Normalize the postings list
            for term in terms:
                tf = log_term_frequencies[term]
                normalised_tf = tf / doc_length
                # check_count += normalised_tf ** 2;

                if term not in postings_lists:
                    postings_lists[term] = []
                # Update the postings list with the document ID and normalised tf
                postings_lists[term].append((int(filename), normalised_tf))  # store the docID + normalised tf in postings list

            # print ('check_count: ', check_count)


    print("Writing...")
    # Write total document number, final dictionary and posting list to respective files
    with open(out_dict, 'w') as dict_file, open(out_postings, 'w') as postings_file:
        # files are opened in write mode to overwrite any existing files
        dict_file.write(f"Total_number_of_document: {len(length)}\n")
        for term in sorted(doc_freq):
            pointer = postings_file.tell()
            for doc in postings_lists[term]:
                postings_file.write(f"{doc}")
                if doc != postings_lists[term][-1]:
                    postings_file.write(", ")
            postings_file.write("\n")
            # postings_file.write(f"{postings_lists[term]}\n")
            dict_file.write(f"{term} {doc_freq[term]} {pointer}\n")




input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i':  # input directory
        input_directory = a
    elif o == '-d':  # dictionary file
        output_file_dictionary = a
    elif o == '-p':  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
