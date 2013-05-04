from collections import Counter
import os
from math import log


class Document(object):
    '''Container class for utility static methods'''
    @staticmethod
    def compute_tf_vector(words):
        tf = {}
        for w in words:
            if w not in tf: tf[w] = 0
            tf[w] += 1
        return tf
    
    @staticmethod
    def compute_tf_norm_vector(words,l):
        l=float(l)
        tf = Document.compute_tf_vector(words)
        for w in tf:
            tf[w] /= l
        return tf
    
    @staticmethod
    def compute_tf_idf_vector(words,corpus = None):
        tf = Document.compute_tf_vector(words)
        if corpus is not None:
            for w in tf:
                tf[w] *= corpus.get_IDF(w)
        return tf
    
    @staticmethod
    def cosine_sim(tf1,tf2):
        tot = 0.0
        for term in [k for k in tf1 if k in tf2]:
            tot += (tf1[term]*tf2[term])
        return tot

class CorpusInfo(object):
    '''Represents a corpus, which can be queried for IDF of a term'''
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
    '''Properties of a single anchor text chunk'''
    def __init__(self,anchor_text,anchor_count):
        self.text = anchor_text
        self.count = anchor_count
     
class Page(object):
    '''Represents a single web page, with all its fields. Contains TF vectors for the fields'''
    def __init__(self,page,page_fields):
        self.url = page
        self.body_length = page_fields.get('body_length',0)
        self.pagerank = page_fields.get('pagerank',0)
        self.title = page_fields.get('title',"")
        self.header = page_fields.get('header',"")
        self.body_hits = page_fields.get('body_hits',0)
        self.anchors = [Anchor(text,count) for text,count in page_fields.get('anchors',{}).iteritems()]
 
        self.field_tf_vectors = self.compute_field_tf_vectors()

    def url_tf_vector(self): # parse/split URL
        pass

    def header_tf_vector(self):
        words = reduce(lambda x,h: x+h.strip().split(),self.header,[])
        return Document.compute_tf_norm_vector(words,self.body_length)

    def body_tf_vector(self):
        tf = {}
        l = float(self.body_length)
        
        for bh in self.body_hits:
            tf[bh] = len(self.body_hits[bh])/l
        
        return tf

    def title_tf_vector(self): # should be same/similar to header method
        words = self.title.strip().split()
        return Document.compute_tf_norm_vector(words,self.body_length)

    def anchor_tf_vector(self):
        pass

    def compute_field_tf_vectors(self):
        tfs = {}
        tfs['url']      = self.url_tf_vector()    # TODO
        tfs['header']   = self.header_tf_vector()
        tfs['body']     = self.body_tf_vector()   
        tfs['title']    = self.title_tf_vector()
        tfs['anchor']   = self.anchor_tf_vector() # TODO
        return tfs
        
# Look in rank0.main() for how this object is created. Also look at the pa3_play ipython notebook.
class Query(object):
    '''A single query, with all the results associated with it'''
    def __init__(self,query,query_pages,corpus=None):  # query_pages : query -> urls
        self.query = query
        self.terms = self.query.strip().split()
        self.pages = dict([(p,Page(p,v)) for p,v in query_pages.iteritems()]) # URLs
        self.tf_vector = Document.compute_tf_idf_vector(self.terms,corpus)

        
