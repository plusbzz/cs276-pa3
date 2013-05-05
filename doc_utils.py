from collections import Counter
import os,sys
from os import path
from math import log
import re
import cPickle as marshal

class Document(object):
    '''Container class for utility static methods'''
    @staticmethod
    def logify(otf):
        tf = {}
        for w in otf:
            tf[w] = (1 + log(otf[w])) if otf[w] > 0 else 0
        return tf
        
    
    @staticmethod
    def compute_tf_vector(words,multiplier = 1):
        tf = {}
        for w in words:
            if w not in tf: tf[w] = 0.0
            tf[w] += 1
        if multiplier > 1:
            for w in tf: tf[w] *= multiplier
        return tf
    
    @staticmethod
    def compute_tf_norm_vector(words,length):
        length=float(length)
        tf = Document.compute_tf_vector(words)
        for w in tf: tf[w] = tf[w]/length
        return tf
    
    @staticmethod
    def IDFy(otf,corpus = None):
        tf = {}
        if corpus is not None:
            for w in otf:
                tf[w] = otf[w]*corpus.get_IDF(w)
        return tf
    
    @staticmethod
    def cosine_sim(tf1,tf2):
        n1 = n2 = 1.0
        tot = 0.0
        # normalize vectors by L1 norm
        if len(tf1) > 0 and len(tf2) > 0:
            n1 = sum([abs(v) for v in tf1.values()])
            n2 = sum([abs(v) for v in tf2.values()])
        
            for term in [k for k in tf1 if k in tf2]:
                tot += (tf1[term]*tf2[term])
        return tot/(n1*n2)

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
        marshal.dump((self.total_file_count,self.df_counter),open("IDF.dat","wb"))
    
    def load_doc_freqs(self):
        if path.isfile("IDF.dat"):
            print >> sys.stderr, "Loading IDF from file"
            self.total_file_count,self.df_counter = marshal.load(open("IDF.dat"))
        else:
            print >> sys.stderr, "Computing IDF"
            self.compute_doc_freqs()
        
    def get_IDF(self,term):
        return log(self.total_file_count) - log(self.df_counter[term]+1.0) # for Laplace smoothing

class Anchor(object):
    '''Properties of a single anchor text chunk'''
    def __init__(self,anchor_text,anchor_count):
        self.text = anchor_text
        self.terms = self.text.lower().strip().split()
        self.count = anchor_count
        self.term_counts = Document.compute_tf_vector(self.terms,self.count)
            
     
class Page(object):
    
    fields = ['url','header','body','anchor','title']
    
    '''Represents a single web page, with all its fields. Contains TF vectors for the fields'''
    def __init__(self,page,page_fields):
        self.url = page
        
        self.body_length = page_fields.get('body_length',1.0)
        self.body_length = (1.0 if self.body_length == 0 else self.body_length)
        
        self.pagerank = page_fields.get('pagerank',0)
        self.title = page_fields.get('title',"")
        self.header = page_fields.get('header',"")
        self.body_hits = page_fields.get('body_hits',{})
        self.anchors = [Anchor(text,count) for text,count in page_fields.get('anchors',{}).iteritems()]
        self.field_tf_vectors = self.compute_field_tf_vectors()

    def url_tf_vector(self): # TODO parse/split URL
        words = filter(lambda x: len(x) > 0,re.split('\W',self.url))
        return Document.compute_tf_norm_vector(words,self.body_length)

    def header_tf_vector(self):
        words = reduce(lambda x,h: x+h.strip().lower().split(),self.header,[])
        return Document.compute_tf_norm_vector(words,self.body_length)

    def body_tf_vector(self):
        tf = {}
        l = float(self.body_length)       
        for bh in self.body_hits:
            tf[bh] = len(self.body_hits[bh])/l       
        return tf

    def title_tf_vector(self): 
        words = self.title.lower().strip().split() # Can do stemming etc here
        return Document.compute_tf_norm_vector(words,self.body_length)

    def anchor_tf_vector(self):
        tf = {}
        for a in self.anchors:
            atf = a.term_counts
            for term in atf:
                if term not in tf: tf[term] = 0.0
                tf[term] += atf[term]
        for term in tf: tf[term] /= self.body_length # normalize
        return tf

    def compute_field_tf_vectors(self):
        tfs = {}
        tfs['url']      = self.url_tf_vector()    # TODO
        tfs['header']   = self.header_tf_vector()
        tfs['body']     = self.body_tf_vector()   
        tfs['title']    = self.title_tf_vector()
        tfs['anchor']   = self.anchor_tf_vector() # TODO
        
        for field in tfs: tfs[field] = Document.logify(tfs[field])
        return tfs
 
class QueryPage(object):  
    field_weights = {
        'url'   :   1.0,
        'header':   1.0,
        'body'  :   1.0,
        'anchor':   1.0,
        'title' :   1.0    
    }
    
    def __init__(self,query,page):
        self.query = query
        self.page = page
        self.field_scores = {}
        self.final_score = 0.0
        self.compute_cosine_scores()
        
    def compute_cosine_scores(self):
        tf1 = self.query.tf_vector
        for field in QueryPage.field_weights:
            tf2 = self.page.field_tf_vectors[field]
            self.field_scores[field] = Document.cosine_sim(tf1,tf2)
            self.final_score += (QueryPage.field_weights[field] * self.field_scores[field])
        
# Look in rank0.main() for how this object is created. Also look at the pa3_play ipython notebook.
class Query(object):

    '''A single query, with all the results associated with it'''
    def __init__(self,query,query_pages,corpus=None):  # query_pages : query -> urls
        self.query = query
        self.terms = self.query.lower().strip().split()
        self.pages = dict([(p,Page(p,v)) for p,v in query_pages.iteritems()]) # URLs
        self.tf_vector = Document.compute_tf_vector(self.terms)
        self.tf_vector = Document.logify(self.tf_vector)
        self.tf_vector = Document.IDFy(self.tf_vector,corpus)
        
    def compute_cosine_scores(self):
        self.page_scores = [QueryPage(self,page) for p,page in self.pages.iteritems()]       
        self.ranked_page_scores = [(qp.page.url,qp.final_score) for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        self.ranked_pages = [qp.page.url for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        return self.ranked_pages        

        
