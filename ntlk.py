import nltk, string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import os


rootdir='E:/2_project/tomcat/tomcat'

# reading the csv file from dataset containing bug reports
myfile = open('small_sample.csv', "r")
reader = csv.reader(myfile)
rows = [[row[1], row[4]] for row in reader]

# reading the file from dataset containign the API specifications
myfile2 = open('test2.csv', "r")
reader2 = csv.reader(myfile2)
rows2 = [[row[1], row[2]] for row in reader2]

# lists to save the results
final_files = []
final_api = []
files_source = []
url_source = []
bug_id = []
bug_id2 = []
id1 = []
id2 = []
br_score = []
code_weight = 14.00
api_weight = 3.69
br_weight = 6.38


# stop words
stops = stopwords.words('english')
#
# finding the stem of the words
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

# stemming anf tokenizing
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

# remove punctuation, lowercase, stem
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words=stops)

# the main function to calculate the cosine similarity between two vectorized text objects
def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

# this is the function to calculate the similarity of bug report to each sourcefile
def sourcecodesimilarity():
    for item in rows:
        for j in item[1:]:
            for subdir, dirs, files in os.walk(rootdir):
                for file in files:
                    if file.endswith(".java"):
                        pathy = os.path.join(subdir, file)
                        myfile = open(pathy,'r')
                        reading = myfile.read()
                        x = cosine_sim(j,reading)
                        x2 = x*code_weight
                        # print(pathy)
                        final_files.append(x2)
                        files_source.append(pathy)
                        for sing in item[:1]:
                            bug_id.append(sing)
#
# this is the function to calculate the similarity of bug reports to API specifications
def apisimilarity():
    for item in rows:
        for j in item[1:]:
            for part in rows2:
                for g in part[1:]:
                    y = cosine_sim(j,g)
                    y2 = y*api_weight
                    final_api.append(y2)
                    z = part[:1]
                    url_source.append(z)
                    for single in item[:1]:
                        bug_id2.append(single)

# this is the function to calculate the collaborating filtering score
def br_scores():
    iterations = range(1,len(rows))
    for item in iterations:
        index = 0
        while(item>index):
            a = rows[item][1]
            b = rows[index][1]
            w = cosine_sim(a,b)
            w2 = w*br_weight
            c= rows[item][0]
            d = rows[index][0]
            id1.append(c)
            id2.append(d)
            br_score.append(w2)
            index=index+1


sourcecodesimilarity()
apisimilarity()
br_scores()


# writing the result to different csv files for further processing
records = zip(id1,id2,br_score)
records2 = zip(bug_id,files_source,final_files)
records3 = zip(bug_id2,url_source,final_api)

with open('br_scores.csv','w', newline='')as f:
    writer=csv.writer(f)
    for record in records:
        writer.writerow(record)

with open('source_code_scores_2.csv','w', newline='')as f:
    writer=csv.writer(f)
    for record in records2:
        writer.writerow(record)

with open('api_scores_2.csv','w', newline='')as f:
    writer=csv.writer(f)
    for record in records3:
        writer.writerow(record)

# print (cosine_sim('''paper introduces an adaptive ranking approach that
# leverages domain knowledge through functional decompositions of source code into methods, API descriptions of
# library components used in the code, the history.''', '''code approach method describe library components'''))
# print (cosine_sim('a little bird', 'a little bird chirps and screw you'))
# print (cosine_sim('a little bird', 'a big dog barks'))