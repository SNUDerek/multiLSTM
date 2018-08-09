#!/usr/bin/python
#### Original Perl Script
# conlleval: evaluate result of processing CoNLL-2000 shared task
# usage:     conlleval [-l] [-r] [-d delimiterTag] [-o oTag] < file
#            README: http://cnts.uia.ac.be/conll2000/chunking/output.html
# options:   l: generate LaTeX output for tables like in
#               http://cnts.uia.ac.be/conll2003/ner/example.tex
#            r: accept raw result tags (without B- and I- prefix;
#                                       assumes one word per chunk)
#            d: alternative delimiter tag (default is single space)
#            o: alternative outside tag (default is O)
# note:      the file should contain lines with items separated
#            by $delimiter characters (default space). The final
#            two items should contain the correct tag and the
#            guessed tag in that order. Sentences should be
#            separated from each other by empty lines or lines
#            with $boundary fields (default -X-).
# url:       http://lcg-www.uia.ac.be/conll2000/chunking/
# started:   1998-09-25
# version:   2004-01-26
# author:    Erik Tjong Kim Sang <erikt@uia.ua.ac.be>
#### Now in Python
# author:    sighsmile.github.io
# version:   2017-05-18
#### tweaks for running in notebook
# author:    snuderek.github.io
# version:   2018-08-xx

from __future__ import division, print_function, unicode_literals
import argparse
import sys
import re
from collections import defaultdict

# sanity check
def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-l", "--latex",
        default=False, action="store_true",
        help="generate LaTeX output"
    )
    argparser.add_argument(
        "-r", "--raw",
        default=False, action="store_true",
        help="accept raw result tags"
    )
    argparser.add_argument(
        "-d", "--delimiter",
        default=None,
        help="alternative delimiter tag (default: single space)"
    )
    argparser.add_argument(
        "-o", "--oTag",
        default="O",
        help="alternative delimiter tag (default: O)"
    )
    args = argparser.parse_args()
    return args



"""
• IOB1: I is a token inside a chunk, O is a token outside a chunk and B is the
beginning of chunk immediately following another chunk of the same Named Entity.
• IOB2: It is same as IOB1, except that a B tag is given for every token, which exists at
the beginning of the chunk.
• IOE1: An E tag used to mark the last token of a chunk immediately preceding another
chunk of the same named entity.
• IOE2: It is same as IOE1, except that an E tag is given for every token, which exists at
the end of the chunk.
• START/END: This consists of the tags B, E, I, S or O where S is used to represent a
chunk containing a single token. Chunks of length greater than or equal to two always
start with the B tag and end with the E tag.
• IO: Here, only the I and O labels are used. This therefore cannot distinguish between
adjacent chunks of the same named entity.

"""
# endOfChunk: checks if a chunk ended between the previous and current word
# arguments:  previous and current chunk tags, previous and current types
# note:       this code is capable of handling other chunk representations
#             than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
#             Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
def endOfChunk(prevTag, tag, prevType, type):
    """
    checks if a chunk ended between the previous and current word;
    arguments:  previous and current chunk tags, previous and current types
    """
    return ((prevTag == "B" and tag == "B") or
        (prevTag == "B" and tag == "O") or
        (prevTag == "I" and tag == "B") or
        (prevTag == "I" and tag == "O") or

        (prevTag == "E" and tag == "E") or
        (prevTag == "E" and tag == "I") or
        (prevTag == "E" and tag == "O") or
        (prevTag == "I" and tag == "O") or

        (prevTag != "O" and prevTag != "." and prevType != type) or
        (prevTag == "]" or prevTag == "["))
        # corrected 1998-12-22: these chunks are assumed to have length 1


# startOfChunk: checks if a chunk started between the previous and current word
# arguments:    previous and current chunk tags, previous and current types
# note:         this code is capable of handling other chunk representations
#               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
#               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
def startOfChunk(prevTag, tag, prevType, type):
    """
    checks if a chunk started between the previous and current word;
    arguments:  previous and current chunk tags, previous and current types
    """
    chunkStart = ((prevTag == "B" and tag == "B") or
        (prevTag == "B" and tag == "B") or
        (prevTag == "I" and tag == "B") or
        (prevTag == "O" and tag == "B") or
        (prevTag == "O" and tag == "I") or

        (prevTag == "E" and tag == "E") or
        (prevTag == "E" and tag == "I") or
        (prevTag == "O" and tag == "E") or
        (prevTag == "O" and tag == "I") or

        (tag != "O" and tag != "." and prevType != type) or
        (tag == "]" or tag == "["))
        # corrected 1998-12-22: these chunks are assumed to have length 1

    #print("startOfChunk?", prevTag, tag, prevType, type)
    #print(chunkStart)
    return chunkStart

def calcMetrics(TP, P, T, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = TP / P if P else 0
    recall = TP / T if T else 0
    FB1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * FB1
    else:
        return precision, recall, FB1

def splitTag(chunkTag, oTag = "O", raw = False):
    """
    Split chunk tag into IOB tag and chunk type;
    return (iob_tag, chunk_type)
    """
    if chunkTag == "O" or chunkTag == oTag:
        tag, type = "O", None
    elif raw:
        tag, type = "B", chunkTag
    else:
        try:
            # split on first hyphen, allowing hyphen in type
            tag, type = chunkTag.split('-', 1)
        except ValueError:
            tag, type  = chunkTag, None
    return tag, type

def countChunks(fileIterator, args=None):
    """
    Process input in given format and count chunks using the last two columns;
    return correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter
    """
    if args is not None:
        boundary = "-X-"     # sentence boundary
        delimiter = args.delimiter
        raw = args.raw
        oTag = args.oTag
    else:
        boundary = "-X-"     # sentence boundary
        delimiter = ' '
        raw = False
        oTag = "O"
        

    correctChunk = defaultdict(int)     # number of correctly identified chunks
    foundCorrect = defaultdict(int)     # number of chunks in corpus per type
    foundGuessed = defaultdict(int)     # number of identified chunks per type

    tokenCounter = 0     # token counter (ignores sentence breaks)
    correctTags = 0      # number of correct chunk tags

    lastType = None # temporary storage for detecting duplicates
    inCorrect = False # currently processed chunk is correct until now
    lastCorrect, lastCorrectType = "O", None    # previous chunk tag in corpus
    lastGuessed, lastGuessedType = "O", None  # previously identified chunk tag

    for line in fileIterator:
        # each non-empty line must contain >= 3 columns
        features = line.strip().split(delimiter)
        if not features or features[0] == boundary:
            features = [boundary, "O", "O"]
        elif len(features) < 3:
             raise IOError("conlleval: unexpected number of features in line %s\n" % line)

        # extract tags from last 2 columns
        guessed, guessedType = splitTag(features[-1], oTag=oTag, raw=raw)
        correct, correctType = splitTag(features[-2], oTag=oTag, raw=raw)

        # 1999-06-26 sentence breaks should always be counted as out of chunk
        firstItem = features[0]
        if firstItem == boundary:
            guessed, guessedType = "O", "O"

        # decide whether current chunk is correct until now
        if inCorrect:
            endOfGuessed = endOfChunk(lastCorrect, correct, lastCorrectType, correctType)
            endOfCorrect = endOfChunk(lastGuessed, guessed, lastGuessedType, guessedType)
            if (endOfGuessed and endOfCorrect and lastGuessedType == lastCorrectType):
                inCorrect = False
                correctChunk[lastCorrectType] += 1
            elif ( endOfGuessed != endOfCorrect or guessedType != correctType):
                inCorrect = False

        startOfGuessed = startOfChunk(lastGuessed, guessed, lastGuessedType, guessedType)
        startOfCorrect = startOfChunk(lastCorrect, correct, lastCorrectType, correctType)
        if (startOfCorrect and startOfGuessed and guessedType == correctType):
            inCorrect = True
        if startOfCorrect:
            foundCorrect[correctType] += 1
        if startOfGuessed:
            foundGuessed[guessedType] += 1

        if firstItem != boundary:
            if correct == guessed and guessedType == correctType:
                correctTags += 1
            tokenCounter += 1

        lastGuessed, lastGuessedType = guessed, guessedType
        lastCorrect, lastCorrectType = correct, correctType

    if inCorrect:
        correctChunk[lastCorrectType] += 1

    return correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter

def evaluate(correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter, latex=False):
    # sum counts
    correctChunkSum = sum(correctChunk.values())
    foundGuessedSum = sum(foundGuessed.values())
    foundCorrectSum = sum(foundCorrect.values())

    # sort chunk type names
    sortedTypes = list(foundCorrect) + list(foundGuessed)
    sortedTypes = list(set([s for s in sortedTypes if s is not None]))
    
    sortedTypes.sort()

    # print overall performance, and performance per chunk type
    if not latex:
        # compute overall precision, recall and FB1 (default values are 0.0)
        precision, recall, FB1 = calcMetrics(correctChunkSum, foundGuessedSum, foundCorrectSum)
        # print overall performance
        print("processed %i tokens with %i phrases; " % (tokenCounter, foundCorrectSum), end='')
        print("found: %i phrases; correct: %i.\n" % (foundGuessedSum, correctChunkSum), end='')
        if tokenCounter:
            print("accuracy: %6.2f%%; " % (100*correctTags/tokenCounter), end='')
            print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                    (precision, recall, FB1))

        for i in sortedTypes:
            precision, recall, FB1 = calcMetrics(correctChunk[i], foundGuessed[i], foundCorrect[i])
            print("%17s: " %i , end='')
            print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                    (precision, recall, FB1), end='')
            print("  %d" % foundGuessed[i])

    # generate LaTeX output for tables like in
    # http://cnts.uia.ac.be/conll2003/ner/example.tex
    else:
        print("        & Precision &  Recall  & F\$_{\\beta=1} \\\\\\hline", end='')
        for i in sortedTypes:
            precision, recall, FB1 = calcMetrics(correctChunk[i], foundGuessed[i], foundCorrect[i])
            print("\n%-7s &  %6.2f\\%% & %6.2f\\%% & %6.2f \\\\" %
                 (i,precision,recall,FB1), end='')
        print("\\hline")

        precision, recall, FB1 = calcMetrics(correctChunkSum, foundGuessedSum, foundCorrectSum)
        print("Overall &  %6.2f\\%% & %6.2f\\%% & %6.2f \\\\\\hline" %
              (precision,recall,FB1))

def conlleval(filename, latex=False):
    correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter = countChunks(open(filename, 'r'))
    evaluate(correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter, latex=latex)
    return
    
        
if __name__ == "__main__":
    args = parse_args()
    # process input and count chunks
    correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter = countChunks(sys.stdin, args)

    # compute metrics and print
    evaluate(correctChunk, foundGuessed, foundCorrect, correctTags, tokenCounter, latex=args.latex)

    sys.exit(0)
