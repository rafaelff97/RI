from collections import Counter
from os import listdir
from os.path import isfile, join
import math
import pickle
import sys
import csv
import gzip
import re
import collections
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from document import Document
import time

class Query:
    def __init__(self,nameFileTsvGz, limitPostings,currentPostings,arrayOpc, sizeLimitWord,listStopWords):
        self.currentPostings = currentPostings
        self.limitPostings = limitPostings
        self.sizeLimitWord = sizeLimitWord
        self.arrayOpc = arrayOpc
        self.listStopWords = listStopWords
        self.nameFileTsvGz = nameFileTsvGz
        self.pathIndexCompletoNormalized = "indexCompleteNormalized"
        self.pathIdfIndexDictionary = "IDFIndex/dictionary.txt"
        self.pathImportantValues = "importantValues/values.txt"
        self.pathDocumentIndex = "documentIndex/documentIndex.txt"
        self.pathQueryResults = "queryResults"
    
    def openAndSearchFile(self, term, file):
        """ Opens a specific file and searches for a specific term returning documents that contain that term """
        file = open( self.pathIndexCompletoNormalized+'/' + file, 'r')
        found = False
        arrayDocuments = []
        while True:
            line = file.readline()
    
            if not line:
                break
    
            values = line.split(",")
    
            if values[0] == term:
                found = True
                arrayDocuments.append(values[1])
            elif found:
                break
        file.close()
        return arrayDocuments

    def searchDocumentsForTerm(self,arrayTerms):
        """ Returns a set of documents that contain any term in the array """
        indexIncomplete = [f for f in listdir( self.pathIndexCompletoNormalized) if isfile(join(self.pathIndexCompletoNormalized, f))]
        arrayResults = []
        for file in indexIncomplete:
            for term in arrayTerms:
                if Document.checkRange(self, term, file):
                    arrayResults += self.openAndSearchFile(term, file)
    
        return set(arrayResults)
    
    
    def termWeight(self,counter):
        """ Calculates the weights of the terms returning a dictionary with this information and the query size """
        queryDictionaryWeight = {}
        with open(self.pathIdfIndexDictionary, "rb") as fp:   # Unpickling
            idfTermDictionary = pickle.load(fp)
    
        for term, count in counter.most_common():
            queryDictionaryWeight[term] = (
                1 + math.log10(count)) * idfTermDictionary[term]
    
        queryLength = sum([math.pow(float(v), 2)
                          for k, v in idfTermDictionary.items() if k in counter.keys()])
        return queryDictionaryWeight, queryLength
    
    def normalizedWeightQuery(self, dictionary):
        total = math.sqrt(sum([math.pow(float(v), 2)
                          for v in dictionary.values()]))
    
        for term in dictionary:
            dictionary[term] = dictionary[term] / total
    
        return dictionary

    def loadTermDocumentNormalized(self, indexedDocuments):
        """ Returns a dictionary with the format: docId: {term:peso} """
        #term, documentId, nVezesQueOTermoApareceNoDocumento, weight
        indexComplete = [f for f in listdir(self.pathIndexCompletoNormalized) if isfile(
            join(self.pathIndexCompletoNormalized, f))]
        table = {}
        for file in indexComplete:
    
            file1 = open((self.pathIndexCompletoNormalized+"/" + file),
                         'r', encoding='utf-8')
    
            while True:
                line = file1.readline()
    
                if not line:
                    break
                values = line.strip().split(",")
    
                if values[1] in indexedDocuments:
                    if values[1] in table:
                        table[values[1]].update({values[0]: values[3]})
                    else:
                        table[values[1]] = {values[0]: values[3]}
            file1.close()
        return table
    
    def score(self,documentsNWeight, queryNWeight):
        """ Calculates the score and returns a dictionary with the format: term: {docid:score} """
        dictResults = {}
    
        for document in documentsNWeight:
            for term in documentsNWeight[document]:
                if term in queryNWeight.keys():
                    score = float(
                        documentsNWeight[document][term]) * queryNWeight[term]
                    if term in dictResults.keys():
                        dictResults[term].update({document: score})
                    else:
                        dictResults[term] = {document: score}
        dictResults = {key: dict(sorted(val.items(), key=lambda ele: ele[1], reverse=True))
                       for key, val in dictResults.items()}
        return dictResults
    
    def countDocumentsForTerm(self,term, file):
        """ Counts the number of times a term in a specific file """
        file = open( self.pathIndexCompletoNormalized+'/' + file, 'r')
        found = False
        counter = 0
        while True:
            line = file.readline()
    
            if not line:
                break
    
            values = line.split(",")
    
            if values[0] == term:
                found = True
                counter += 1
            elif found:
                break
        file.close()
        return counter

    def checkRange(self,term, rangeFicheiro):
        """ Checks if a term belongs to a file """
        splittedRange = rangeFicheiro.split("-")
        splittedRange.append(term)
        organizedTerms = sorted(splittedRange)
    
        if term == organizedTerms[1]:
            return True
        else:
            return False
        
    def tfi(self,term):
        """ Counts the number of documents that contain a specific term """
        indexIncompleto = [f for f in listdir( self.pathIndexCompletoNormalized) if isfile(
            join( self.pathIndexCompletoNormalized, f))]
        totalDocumentsForTerm = 0
        for file in indexIncompleto:
            if self.checkRange(term, file):
                totalDocumentsForTerm = self.countDocumentsForTerm(term, file)
    
        return totalDocumentsForTerm
    
    def bm25(self, k1, b, query, documents):
        """ Does the maths for bm25 """
        dictDocumentBM25 = {}
        dictTFI = {}
        for term in query:
            dictTFI[term] = self.tfi(term)
    
        documentTermDictionary = self.loadTermDocumentNormalized(documents)
    
        with open(self.pathIdfIndexDictionary, "rb") as fp:   # Unpickling
            idfTermDictionary = pickle.load(fp)
    
        f = open(self.pathImportantValues, "r")
        mediaLenDocuments = float(f.readline().split(",")[0])
        f.close()
    
        f = open(self.pathDocumentIndex, "r")
        content = f.readlines()
        for documentIndex in documents:
            bm25Document = 0
            lenDocument = content[int(documentIndex)].split(",")[2]
    
            for term in query:
                if term in documentTermDictionary[documentIndex].keys():
    
                    mediaDocument = float(lenDocument)/mediaLenDocuments
                    tfTerm = dictTFI[term]
                    bm25Document += idfTermDictionary[term] * (((k1 + 1) * tfTerm)) / (
                        k1 * ((1 - b) + b * mediaDocument) + tfTerm)
    
                    if term in dictDocumentBM25.keys():
                        dictDocumentBM25[term].update(
                            {documentIndex: bm25Document})
                    else:
                        dictDocumentBM25[term] = {documentIndex: bm25Document}
    
        dictDocumentBM25 = {key: dict(sorted(val.items(), key=lambda ele: ele[1], reverse=True))
                            for key, val in dictDocumentBM25.items()}
        f.close()
        return dictDocumentBM25
    
    def printResultQuery(self, documentIndex, query):
        """ Prints the query results in a more organised manner """
        if len(documentIndex) == 0:
            with open(self.pathQueryResults+"/"+query+".txt", 'a+') as f:
                f.write("0")
            f.close()
        else:
            with open(self.pathIdfIndexDictionary, "rb") as fp:   # Unpickling
                idfTermDictionary = pickle.load(fp)
                
            with open(self.pathQueryResults+"/"+query+".txt", 'a+') as f:
        
                termInfo = ""
                for term in documentIndex:
                    termInfo += str(term) + ":" + \
                        str(round(idfTermDictionary[term], 2)) + ";"
                    for document in documentIndex[term]:
                        termInfo += str(document) + ":" + \
                            str(round(documentIndex[term][document], 2)) + ";"
                    print(termInfo)
                    f.write(termInfo)
                    f.write("\n")
                    termInfo = ""
            f.close()
    
    def documentIndexToDocumentId(self, indexedDocuments):
        """ Returns a dictionary where the temporary id of the document is the original """
        resultDict = {}
        for term in indexedDocuments:
            top100 = 0
            for document in indexedDocuments[term]:
                with open(self.pathDocumentIndex) as f:
                    data = f.readlines()[int(document)]
                f.close()
                if term in resultDict.keys():
                    resultDict[term].update(
                        {data.split(",")[1]: indexedDocuments[term][document]})
                else:
                    resultDict[term] = {data.split(
                        ",")[1]: indexedDocuments[term][document]}
                top100 += 1
                if top100 == 100:
                    break
    
        return resultDict
    
    def cleaner(self,text):
        """ Erase symbols """
        emoji_pattern = re.compile("[^\w']|_", flags=re.UNICODE)
        return (emoji_pattern.sub(" ", text))

    def minimumLength(self,array, size):
        """ Returns an array whose terms are larger than the specified size """
        new_list = []
        for e in array:
            if len(e) > size:
                new_list.append(e)
        return new_list
    
    def defaultListSW(self,array):
        """ Returns an array of all terms that do not belong to the stopwords list """
        stops = set(stopwords.words('english'))
        arrayTemp = []
        #print("Lista de stop words:"+str(stops))
        for line in array:
            for w in line.split():
                if w.lower() not in stops:
                    arrayTemp.append(w)
        return arrayTemp
    
    def userDefined(self,array, listt):
        """user defined StopWords"""
        arrayTemp = []
        stopWordsList = listt.split()
        for line in array:
            for w in line.split():
                if w.lower() not in stopWordsList:
                    arrayTemp.append(w)
        return arrayTemp
    
    def defaultListPS(self,array):
        ps = PorterStemmer()
        arrayTemp = []
        for w in array:
            arrayTemp.append(ps.stem(w))
        return arrayTemp
    
    
    def snowBall(self,array):
        snow_stemmer = SnowballStemmer(language='english')
        stem_words = []
        for w in array:
            x = snow_stemmer.stem(w)
            stem_words.append(x)
    
        return stem_words
    def fillTableAnalise(self, tokenizedText, table):
        """ Update a dictionary of terms:counter with tokenizedText """
        uniqueTerms = set(tokenizedText)
    
        for term in uniqueTerms:
    
            if term in table:
                table[term] += 1
            else:
                table[term] = 1
    
        return table
    
    def tokenizer(self,*args):
            textoArray = []
            for text in args:
                text = self.cleaner(text)
                textoArray += text.split()
        
            if self.arrayOpc[0] == '1':
                textoArray = self.minimumLength(textoArray, self.sizeLimitWord)
            elif self.arrayOpc[0] == '2':
                textoArray = textoArray
        
            if self.arrayOpc[1] == '1':
                textoArray = textoArray
            elif self.arrayOpc[1] == '2':
                textoArray = self.defaultListSW(textoArray)
            elif self.arrayOpc[1] == '3':
                textoArray = self.userDefined(textoArray, self.listStopWords)
        
            if self.arrayOpc[2] == '1':
                textoArray = textoArray
            elif self.arrayOpc[2] == '2':
                textoArray = self.defaultListPS(textoArray)
            elif self.arrayOpc[2] == '3':
                textoArray = self.snowBall(textoArray)
            return textoArray
        
    
    def query(self):
        """ Method in charge of doing all the calculations and showing the query results """
        t0 = time.time()
        query = str(input("What is your query:\n"))

        method = str(input("Pick your prefered method to run:\n" +
                       "1 -> lnc-ltc\n" +
                       "2 -> bm25\n"))        

        queryTokenize = self.tokenizer(query)
        counts = Counter(queryTokenize)
        dictionaryTermWeight, queryLength = self.termWeight(counts)  
        finalDict={}
        indexedDocuments = self.searchDocumentsForTerm(queryTokenize)
        flag = ""
        if method == "1":
            dictionaryTermNormalizedWeight = self.normalizedWeightQuery(
                dictionaryTermWeight)
            dictDocumentsNormalized = self.loadTermDocumentNormalized(indexedDocuments)
            finalDict = self.score(dictDocumentsNormalized,
                              dictionaryTermNormalizedWeight)
            flag = "lncltc"
        elif method == "2":
            #k1 = float(input("What is the value of k1:\n"))
            #b = float(input("What is the value of b:\n"))
            #if k1<=0 or k1 == "": or b<=0:
            k1=1.2
            b=0.75
            finalDict = self.bm25(k1, b, queryTokenize, indexedDocuments)
            flag="bm25"

        documentScoreDict = self.documentIndexToDocumentId(finalDict)
    
        self.printResultQuery(documentScoreDict, str(query+"_"+flag))
        t1=time.time()
        print("Tempo que demora a fazer um search: "+str(t1-t0))
    
    def analysesComplete(self,table):
        """ Split the table into ranges of x postings, the result is to be used later in the merging """
        postings = self.limitPostings
        splitResults = []
        firstTerm = ""
    
        for key, value in table.items():
            while value != 0:
                if firstTerm == "":
                    firstTerm = key
    
                postings -= value
                value = 0
    
                if postings <= 0:
                    splitResults.append(str(firstTerm + "-" + key))
                    firstTerm = ""
                    postings = self.limitPostings
    
        if postings > 0:
            splitResults.append(str(firstTerm + "-" + key))
    
        print(splitResults)
    
    def analyses(self):
        """ Read the file and fill in the table with the terms:counter and then do the analysis to know how best to separate by file """
        maxInt = sys.maxsize
        while True:
            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt/10)
    
        with gzip.open(self.nameFileTsvGz, 'rt', encoding='utf-8') as f:
            tsv_reader = csv.reader(f, delimiter="\t")
            line_count = 0
            table = {}
    
            for row in tsv_reader:
                if line_count == 0:
                    line_count += 1
                else:       
                    try:
                        tokenizerText = self.tokenizer(row[5], row[13], row[12])
                    except:
                      print("One of the rows had an error so it was skipped!")
                    table = self.fillTableAnalise(tokenizerText, table)
    
        # for key in tabela:
            #print(str(key[0]) + " - " + str(tabela[key[1]]))
        # print(collections.OrderedDict(sorted(tabela.items())))
        self.analysesComplete(collections.OrderedDict(sorted(table.items())))
