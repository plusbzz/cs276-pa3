import sys
import re
from math import log
from doc_utils import *
from random import randint
import ndcg

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
      
    f.close()
    return (queries, features) 

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features):
    rankedQueries = {}
    for query in queries.keys():
      results = queries[query]
      #features[query][x].setdefault('body_hits', {}).values() returns the list of body_hits for all query terms
      #present in the document, empty if nothing is there. We sum over the length of the body_hits array for all
      #query terms and sort results in decreasing order of this number
      rankedQueries[query] = sorted(results, 
                                    key = lambda x: sum([len(i) for i in 
                                    features[query][x].setdefault('body_hits', {}).values()]), reverse = True)

    return rankedQueries


#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries,outFileName):
    with open(outFileName,"wb") as outfile:
        for query in queries:
            print >> outfile, ("query: " + query)
            for res in queries[query]:
                print >> outfile, ("  url: " + res)


def cosineRankQueries_withSmallestWindow(features,corpus = None):
    useSmallestWindow = True
    return dict([(query,Query(query,features[query],corpus).compute_cosine_scores(useSmallestWindow)) for query in features])
    
#inparams
#  featureFile: file containing query and url features
def main(featureFile):
    #output file name
    outputFileName = "ranked.txt" #Please don't change this!

    print >> sys.stderr, "Analyzing corpus for IDF info"
    corpus = CorpusInfo("data")
    corpus.load_doc_freqs()
    
    #populate map with features from file
    (queries, features) = extractFeatures(featureFile)
    
    QueryPage.cosine_w = {
        'url'   :   -1705,
        'header':   -85,
        'body'  :   -97,
        'anchor':   -17,
        'title' :   -288
    }
    
    QueryPage.smallest_window_boost = 6.0
  
    #calling baseline ranking system, replace with yours
    rankedQueries = cosineRankQueries_withSmallestWindow(features,corpus)
    
    #print ranked results to file
    printRankedResults(rankedQueries,outputFileName)
    
    print >> sys.stderr, "Score: ", ndcg.scoreResults(outputFileName,'queryDocTrainRel')
    #print >> sys.stderr, "Score: ", ndcg.scoreResults(outputFileName,'queryDocTrainRelDev')        
    
    ## run experiments to determine best cosine weights
    #best_weights = QueryPage.cosine_w
    #best_score = 0.0
    #
    #for i in xrange(1,10000):
    #    QueryPage.cosine_w = {
    #        'url'   :   randint(-2500,500),
    #        'header':   randint(-100,0),
    #        'body'  :   randint(-500,0),
    #        'anchor':   randint(-50,500),
    #        'title' :   randint(-500,0)    
    #    }
    #    rankedQueries = cosineRankQueries(features,corpus)
    #    printRankedResults(rankedQueries,outputFileName)
    #    score = ndcg.scoreResults(outputFileName,'queryDocTrainRel')        
    #    if score > best_score:
    #        best_score = score
    #        best_weights = QueryPage.cosine_w
    #        print >> sys.stderr, "New best:",best_score,best_weights
    #    if i%100 == 0: print >> sys.stderr,"Number of trials:",i,QueryPage.cosine_w
        
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    main(sys.argv[1])
