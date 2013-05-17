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
            print >> outfile, ("query: " + query)
            for res in queries[query]:
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
    QueryPageBM25F.bm25f_K1    = 1.5
    QueryPageBM25F.lamd        = 1.0
    QueryPageBM25F.lamd_prime  = 1.0
    QueryPageBM25F.lamd_prime2 = 1.0
    
    QueryPageBM25F.Vf = v_logarithmic
    fields_avg_len    = DocUtils.features_avg_len(features)
  
    #calling baseline ranking system, replace with yours
    rankedQueries = bm25fRankQueries(features,fields_avg_len,corpus)
        
    #print ranked results to file
    printRankedResults(rankedQueries,outputFileName)
    
    print >> sys.stderr, "Score: ", ndcg.scoreResults(outputFileName,'queryDocTrainRel')    
    
    ## run experiments to determine best cosine weights
    #best_B           = QueryPageBM25F.bm25f_B
    #best_W           = QueryPageBM25F.bm25f_W
    #best_K1          = QueryPageBM25F.K1
    #best_lamd        = QueryPageBM25F.lamd
    #best_lamd_prime  = QueryPageBM25F.lamd_prime
    #best_lamd_prime2 = QueryPageBM25F.lamd_prime2
    #best_score       = 0.0
    #
    #for i in xrange(1,10000):
    #    # [url,title,header,body,anchor]
    #    QueryPageBM25F.bm25f_B     = [uniform(0.0,1.0),uniform(0.0,1.0),uniform(0.0,1.0),uniform(0.0,1.0),uniform(0.0,1.0)]
    #    QueryPageBM25F.bm25f_W     = [uniform(0.0,1.0),uniform(0.0,1.0),uniform(0.0,1.0),uniform(0.0,1.0),uniform(0.0,1.0)]
    #    QueryPageBM25F.bm25f_K1    = uniform(1.2,2.0)
    #    QueryPageBM25F.lamd        = uniform(0.0,0.1)
    #    QueryPageBM25F.lamd_prime  = uniform(0.0,0.1)
    #    QueryPageBM25F.lamd_prime2 = uniform(0.0,0.1)
    #    
    #    rankedQueries = bm25fRankQueries(features,fields_avg_len,corpus)
    #    printRankedResults(rankedQueries,outputFileName)
    #    score = ndcg.scoreResults(outputFileName,'queryDocTrainRel')
    #    
    #    if score > best_score:
    #        best_score       = score
    #        best_B           = QueryPageBM25F.bm25f_B
    #        best_W           = QueryPageBM25F.bm25f_W
    #        best_K1          = QueryPageBM25F.K1
    #        best_lamd        = QueryPageBM25F.lamd
    #        best_lamd_prime  = QueryPageBM25F.lamd_prime
    #        best_lamd_prime2 = QueryPageBM25F.lamd_prime2
    #        print >> sys.stderr, "New best:",best_score,best_B,best_W,best_K1,best_lamd,best_lamd_prime,best_lamd_prime2
    #        
    #    if i%100 == 0: print >> sys.stderr,"Number of trials: ",i,QueryPageBM25F.bm25f_B,QueryPageBM25F.bm25f_W,QueryPageBM25F.K1,QueryPageBM25F.lamd,QueryPageBM25F.lamd_prime,QueryPageBM25F.lamd_prime2
    #
    #print >> sys.stderr, "Final best:",best_score,best_B,best_W,best_K1,best_lamd,best_lamd_prime,best_lamd_prime2
        
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    main(sys.argv[1])
