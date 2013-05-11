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


def avgLen(features):
    ''' Gets the average length of each each field (body, url, title, header, anchor)'''
        
    body_length = 'body_length'
    title       = 'title'
    header      = 'header'
    anchor      = 'anchors'
    url         = 'url'
    
    # (total_count, total_sum)
    body_counts   = [0,0] 
    title_counts  = [0,0]
    header_counts = [0,0]
    anchor_counts = [0,0]
    url_counts    = [0,0]
    
    for query in features:
        for url in features[query]:
            process_body(body_length, body_counts, features[query][url])
            process_title(title, title_counts, features[query][url])
            process_header(header, header_counts, features[query][url])
            process_anchor(anchor, anchor_counts, features[query][url])
            process_url(url_counts,url)

    return {'body':   body_counts[1]   / float(body_counts[0]) if body_counts[0] != 0 else 0.0,
            'url':    url_counts[1]    / float(url_counts[0]) if url_counts[0] != 0 else 0.0,
            'title':  title_counts[1]  / float(title_counts[0]) if title_counts[0] != 0 else 0.0,
            'header': header_counts[1] / float(header_counts[0]) if header_counts[0] != 0 else 0.0,
            'anchor': anchor_counts[1] / float(anchor_counts[0]) if anchor_counts[0] != 0 else 0.0}
    
def process_url(url_counts, url_content):
    url_terms      = filter(lambda x: len(x) > 0, re.split('\W',url_content))
    url_counts[0] += 1
    url_counts[1] += len(url_terms)
    
def process_body(body_length, body_counts, url):
    body_counts[0] += 1
    if body_length in url:
        body_counts[1] += int(url[body_length])
        

def process_title(title, title_counts, url):
    title_counts[0] += 1
    if title in url:
        title_counts[1] += len(url[title].strip().split())

def process_header(header, header_counts, url):
    if header in url:
        header_content    = url[header] # header_content is a List of Strings
        header_counts[0] += len(header_content)     
        header_counts[1] += sum([len(header.strip().split()) for header in header_content])
    else:
        header_counts[0] += 1

def process_anchor(anchor, anchor_counts, url):
    if anchor in url:
        anchor_content = url[anchor] # anchor_content is a dictionary with key=anchor_text, value=stanford_anchor_count
        anchor_counts[0] += len(anchor_content)
        anchor_counts[1] += sum([len(anchor.strip().split()) for anchor in anchor_content])
    else:
        anchor_counts[0] +=1


def cosineRankQueries(features,corpus = None):
    return dict([(query,Query(query,features[query],corpus).compute_cosine_scores()) for query in features])

def bm25fRankQueries(features,corpus = None):
    return dict([(query,QueryBM25F(query,features[query],corpus).compute_bm25f_scores()) for query in features])
    
    #return dict([("query1",["url1","url2","url3"]),("query2",["url4","url5","url6"]),("query3",["url7","url8","url9"])])

    
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
  
    #calling baseline ranking system, replace with yours
    rankedQueries = bm25fRankQueries(features,corpus)
    
    #print ranked results to file
    printRankedResults(rankedQueries,outputFileName)
    
        
if __name__=='__main__':
    if (len(sys.argv) < 2):
      print "Insufficient number of arguments" 
    main(sys.argv[1])
