import sys
import re
from math import log,exp
from doc_utils import *
from random import randint,uniform
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
            print("query: " + query)
            print >> outfile, ("query: " + query)
            for res in queries[query]:
                print("  url: " + res)                
                print >> outfile, ("  url: " + res)

def cosineRankQueries(features,corpus = None):
    return dict([(query,Query(query,features[query],corpus).compute_cosine_scores()) for query in features])

def bm25fRankQueries(features, features_avg_len, corpus):
    return dict([(query,QueryBM25F(query,features[query],features_avg_len, corpus).compute_bm25f_scores()) for query in features])

def v_logarithmic(self, lamda_prime, field, lamda_prime2=1):
    value = lamda_prime + field
    return log(value) if value > 0 else 0

def v_saturation(self, lamda_prime, field, lamda_prime2=1):
    den   = lamda_prime + field
    return (float(field) / den) if den > 0 else 0 

def v_sigmoid(self, lamda_prime, field, lamda_prime2=1):
    den = lamda_prime + exp(-field*lamda_prime2)
    return (1.0 / den) if den > 0  else 0
    
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
    
    # [url,title,header,body,anchor]
    QueryPageBM25F.bm25f_B     = [1.0,0.1,1.0,1.0,0.1]
    QueryPageBM25F.bm25f_W     = [1.0,0.9,0.8,0.9,0.7]
    QueryPageBM25F.K1          = 1
    QueryPageBM25F.lamd        = 3.0
    QueryPageBM25F.lamd_prime  = 2.0
    QueryPageBM25F.lamd_prime2 = 1.0
    
    QueryPageBM25F.Vf = v_logarithmic
    #QueryPageBM25F.Vf = v_sigmoid
    #QueryPageBM25F.Vf = v_saturation
    fields_avg_len    = DocUtils.features_avg_len(features)
  
    #calling baseline ranking system, replace with yours
    rankedQueries = bm25fRankQueries(features,fields_avg_len,corpus)
        
    #print ranked results to file
    printRankedResults(rankedQueries,outputFileName)
            
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    main(sys.argv[1])
