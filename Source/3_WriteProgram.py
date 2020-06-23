#Import Packages
import nltk
from collections import Counter
from nltk import sent_tokenize

#Open file 'inputdata.txt'
fileName = 'inputdata.txt'
file = open(fileName,'r')
#Read from file and save in string 'contents'
contents = file.read()
#Create a lemmatizer and tokenizer from nltk
lmtzr = nltk.WordNetLemmatizer()
t = nltk.WordPunctTokenizer()
#Break string into tokens and save to 'tok'
tok = t.tokenize(str(contents))
#Lemmatize the tokenized data from contents and print
lem = [lmtzr.lemmatize(t) for t in tok]
print("\nLemmatization on words from 'inputdata.txt':\n",lem)

#Break lemmatized/tokenized data into bigrams
bigram = nltk.bigrams(tok)
#Save bigrams in printable version and print them
bigram_print = [i for i in bigram]
print("\nBigram of contents from 'inputdata.txt':\n",bigram_print)
#Count frequency of bigrams and print
bigram_counter = Counter(bigram_print)
print("\nFrequency of bigrams in 'inputdata.txt':\n",bigram_counter)
#Choose top five most repeated bigrams and print
bigram_top5 = bigram_counter.most_common(5)
print("\nTop 5 most repeated bigrams:\n",bigram_top5)

#Separate contents from 'inputdata.txt' into sentences
sentences = sent_tokenize(contents)
#Create an empty array for concatenated sentences
new_sent = []
#Traverse through sentences
for sentence in sentences:
    #Create a temp array to store top 5 bigrams, and new bigrams from each sentence. This will be used to determine
    #if a top 5 bigram is in this sentence
    temp = []
    #Store bigrams from top 5 in temp array
    for x in bigram_top5:
        temp.append(x)
    #Separate sentence into tokens
    sent_tok = t.tokenize(sentence)
    #Break tokenized sentence into bigrams
    sent_bigram = [i for i in nltk.bigrams(sent_tok)]
    #Add new bigrams into temp array
    for x in sent_bigram:
        temp.append(x)

    #Counts the number of bigrams in temp
    num = Counter(temp)
    #If count is more than one, then the bigram is in this sentence and add to new array new_sent
    for items, keys in num.items():
        if keys > 1:
            if sentence not in new_sent:
                new_sent.append(sentence)

#Print all sentences that include bigrams from top 5
print("\nConcatenated sentences:")
for x in new_sent:
    print(x)


