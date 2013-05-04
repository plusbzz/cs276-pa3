from collections import Counter
import os
from math import log

class CorpusInfo(object):
    def __init__(self,corpus_root_dir): # for Laplace smoothing
        self.corpus_dir = corpus_root_dir
        self.total_file_count = 1.0
        self.df_counter = Counter()   # term -> doc_freq
        
    def compute_doc_freqs(self):
        root = self.corpus_dir
        for d in sorted(os.listdir(root)):
          print >> sys.stderr, 'processing dir: ' + d
          dir_name = os.path.join(root, d) 
          term_doc_list = []
          
          for f in sorted(os.listdir(dir_name)):
            self.total_file_count += 1
            
            # Add 'dir/filename' to doc id dictionary
            file_name = os.path.join(d, f)

            fullpath = os.path.join(dir_name, f)
            
            with open(fullpath, 'r') as infile:
                lines = [line for line in infile.readlines()]
                tokens = set(reduce(lambda x,line: x+line.strip().split(),lines,[]))   
                for token in tokens: self.df_counter[token] += 1
    
    def get_IDF(self,term):
        return log(self.df_counter[term]+1.0) - log(self.total_file_count) # for Laplace smoothing

class Anchor(object):
    def __init__(self,anchor_text,anchor_count):
        self.text = anchor_text
        self.count = anchor_count
 
class Document(object):
    def compute_tf_vector(self,words):
        tf = {}
        for w in words:
            if w not in tf: tf[w] = 0
            tf[w] += 1
        return tf

    def compute_tf_idf_vector(self,words,corpus):
        tf = self.compute_tf_vector(words)
        for w in tf:
            tf[w] *= corpus.get_IDF(w)
        
class Page(Document):
    def __init__(self,page,page_fields):
        self.url = page
        self.body_length = page_fields.get('body_length',0)
        self.pagerank = page_fields.get('pagerank',0)
        self.title = page_fields.get('title',"")
        self.body_hits = page_fields.get('body_hits',0)
        self.anchors = [Anchor(text,count) for text,count in page_fields.get('anchors',{}).iteritems()]
 
        self.tf_vectors = {}
        
    def compute_tf_vectors(self):
        pass
         
        
class Query(Document):
    def __init__(self,query,query_pages,corpus):  # query_pages : query -> urls
        self.query = query
        self.terms = self.query.strip().split()
        self.pages = dict([(p,Page(p,v)) for p,v in query_pages.iteritems()]) # URLs
        self.tf_vector = self.compute_tf_idf_vector(self.terms,corpus)

        
