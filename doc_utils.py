from collections import Counter
import os,sys
from os import path
from math import log
import re
import cPickle as marshal

class DocUtils(object):
    '''Container class for utility static methods'''
    LOGIFY = True
    NORMALIZE = True
    
    @staticmethod
    def calculateRawFrequencies(term, queryPages):
        return dict([(url, page.compute_raw_tf_vector(term)) for url,page in queryPages.iteritems()])
            
    @staticmethod
    def compute_bm25f_rawtf_map(queryTerms, queryPages):
        """
        Retuns a map from queryTerm to queryFrequency
        queryFrequency is a map from url to rawFrequencies
        rawFrequencies is a list [term_raw_frequency_url, term_raw_frequency_title, term_raw_frequency_header, term_raw_frequency_body, term_raw_frequency_anchor]
        """
        return dict([ (term, DocUtils.calculateRawFrequencies(term,queryPages)) for term in queryTerms])
    
    @staticmethod
    def logify(otf):
        if not DocUtils.LOGIFY: return otf
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
    def IDFy(otf,corpus = None):
        tf = {}
        if corpus is not None:
            for w in otf:
                tf[w] = otf[w]*corpus.get_IDF(w)
        return tf
    
    @staticmethod
    def normalize(otf,length):
        if not DocUtils.NORMALIZE: return otf
        length=float(length)
        tf = {}
        for w in otf:
            tf[w] = otf[w]/length
        return tf
            
    @staticmethod
    def cosine_sim(tf1,tf2):
        norm = 1.0
        tot = 0.0
        # normalize vectors by L1 norm
        if len(tf1) > 0 and len(tf2) > 0:
            n1 = sum([abs(v) for v in tf1.values()])
            n2 = sum([abs(v) for v in tf2.values()])
            norm = n1*n2
            for term in [k for k in tf1 if k in tf2]:
                tot += (tf1[term]*tf2[term])
        return (tot/norm) if norm > 0 else 0
    
    @staticmethod          
    def url_tf_vector(url): # TODO parse/split URL
        words = filter(lambda x: len(x) > 0,re.split('\W',url))
        return DocUtils.compute_tf_vector(words)

    @staticmethod
    def header_tf_vector(header):
        words = reduce(lambda x,h: x+h.strip().lower().split(),header,[])
        return DocUtils.compute_tf_vector(words)

    @staticmethod
    def body_tf_vector(body_hits):
        tf = {}
        for bh in body_hits:
            tf[bh] = len(body_hits[bh])     
        return tf

    @staticmethod
    def title_tf_vector(title): 
        words = title.lower().strip().split() # Can do stemming etc here
        return DocUtils.compute_tf_vector(words)

    @staticmethod
    def anchor_tf_vector(anchors):
        tf = {}
        for a in anchors:
            atf = a.term_counts
            for term in atf:
                if term not in tf: tf[term] = 0.0
                tf[term] += atf[term]
        return tf
    
    @staticmethod
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
                process_body_length(body_length, body_counts, features[query][url])
                process_title_length(title, title_counts, features[query][url])
                process_header_length(header, header_counts, features[query][url])
                process_anchor_length(anchor, anchor_counts, features[query][url])
                process_url_length(url_counts,url)
    
        return {'body':    body_counts[1]   / float(body_counts[0]) if body_counts[0] != 0 else 0.0,
                'url':     url_counts[1]    / float(url_counts[0]) if url_counts[0] != 0 else 0.0,
                'title':   title_counts[1]  / float(title_counts[0]) if title_counts[0] != 0 else 0.0,
                'headers': header_counts[1] / float(header_counts[0]) if header_counts[0] != 0 else 0.0,
                'anchors': anchor_counts[1] / float(anchor_counts[0]) if anchor_counts[0] != 0 else 0.0}
        
    @staticmethod        
    def process_url_length(url_counts, url_content):
        url_terms      = filter(lambda x: len(x) > 0, re.split('\W',url_content))
        url_counts[0] += 1
        url_counts[1] += len(url_terms)
        
    @staticmethod        
    def process_body_length(body_length, body_counts, url):
        body_counts[0] += 1
        if body_length in url:
            body_counts[1] += int(url[body_length])
            
    @staticmethod    
    def process_title_length(title, title_counts, url):
        title_counts[0] += 1
        if title in url:
            title_counts[1] += len(url[title].strip().split())
    
    @staticmethod    
    def process_header_length(header, header_counts, url):
        if header in url:
            header_content    = url[header] # header_content is a List of Strings
            header_counts[0] += len(header_content)     
            header_counts[1] += sum([len(header.strip().split()) for header in header_content])
        else:
            header_counts[0] += 1
    
    @staticmethod    
    def process_anchor_length(anchor, anchor_counts, url):
        if anchor in url:
            anchor_content = url[anchor] # anchor_content is a dictionary with key=anchor_text, value=stanford_anchor_count
            anchor_counts[0] += len(anchor_content)
            anchor_counts[1] += sum([len(anchor.strip().split()) for anchor in anchor_content])
        else:
            anchor_counts[0] +=1
    

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
        self.term_counts = DocUtils.compute_tf_vector(self.terms,self.count)
            
     
class Page(object):
    
    fields = ['url','header','body','anchor','title']
    
    '''Represents a single web page, with all its fields. Contains TF vectors for the fields'''
    def __init__(self,page,page_fields):
        self.url = page
        
        self.body_length = page_fields.get('body_length',1.0)
        self.body_length = max(1000.0,self.body_length) #(500.0 if self.body_length == 0 else self.body_length)
        
        self.pagerank = page_fields.get('pagerank',0)
        self.title = page_fields.get('title',"")
        self.header = page_fields.get('header',"")
        self.body_hits = page_fields.get('body_hits',{})
        self.anchors = [Anchor(text,count) for text,count in page_fields.get('anchors',{}).iteritems()]
        self.field_tf_vectors = self.compute_field_tf_vectors()
        
        # Combine field vectors
        self.tf_vector = {}       
        for field in QueryPage.cosine_w:
            tf_vec = self.field_tf_vectors[field]
            for term in tf_vec:
                if term not in self.tf_vector:
                    self.tf_vector[term] = 0.0
                self.tf_vector[term] += (QueryPage.cosine_w[field] * tf_vec[term])
                

    def compute_field_tf_vectors(self):
        tfs = {}
        tfs['url']      = DocUtils.url_tf_vector(self.url)
        tfs['header']   = DocUtils.header_tf_vector(self.header)
        tfs['body']     = DocUtils.body_tf_vector(self.body_hits)   
        tfs['title']    = DocUtils.title_tf_vector(self.title)
        tfs['anchor']   = DocUtils.anchor_tf_vector(self.anchors)
        
        for field in tfs: tfs[field] = DocUtils.normalize(tfs[field],self.body_length)
        for field in tfs: tfs[field] = DocUtils.logify(tfs[field])
        return tfs

class PageBM25F(object):
    
    fields = ['url','header','body','anchor','title']
    
    '''Represents a single web page, with all its fields. Contains TF vectors for the fields'''
    def __init__(self,page,page_fields):
        self.url         = page
        self.url_content = filter(lambda x: len(x) > 0, re.split('\W', self.url))
        
        
        self.body_length = page_fields.get('body_length',1.0)
        self.body_length = max(1000.0,self.body_length) #(500.0 if self.body_length == 0 else self.body_length)
        
        self.title         = page_fields.get('title',"")
        self.title_content = self.title.strip().split()
        
        self.headers         = page_fields.get('header',[])
        self.headers_content = [ item for header in self.headers for item in header.strip().split() ]
        
        self.anchors         = page_fields.get('anchors',{})
        self.anchors_content = [ item for anchor in self.anchors for item in anchor.strip().split() ]
        
        self.body_hits = page_fields.get('body_hits',{})
        self.pagerank = page_fields.get('pagerank',0)
        
        self.field_length = {'url':len(self.url_content),
                             'title':len(self.title_content),
                             'headers':len(self.headers_content),
                             'body':self.body_length,
                             'anchors':len(self.anchors_content)}
        
    def raw_tf_in_url(self, term):
        return self.url_content.count(term)
    
    def raw_tf_in_title(self, term):
        return self.title_content.count(term)
    
    def raw_tf_in_headers(self, term):
        return self.headers_content.count(term)
    
    def raw_tf_in_body(self, term):
        term_postings = self.body_hits.get(term,[])
        return len(term_postings)
    
    def raw_tf_in_anchors(self, term):
        return self.anchors_content.count(term)

    def compute_raw_tf_vector(self, term):
        """
        Returns raw frequency of term in each field of the page
        [raw_tf_url, raw_tf_title, raw_tf_header, raw_tf_body, raw_tf_anchor]
        """
        raw_tfs = []
        raw_tfs.append(self.raw_tf_in_url(term))
        raw_tfs.append(self.raw_tf_in_title(term))
        raw_tfs.append(self.raw_tf_in_headers(term))
        raw_tfs.append(self.raw_tf_in_body(term))
        raw_tfs.append(self.raw_tf_in_anchors(term))
 
        return raw_tfs


class DocInfo(object):
    '''Class for non-query-relevant aspects of pages'''
    pages = {}
    
    @staticmethod
    def add_page(page):
        if page.url not in pages:
            pass
            
class QueryPage(object):  
    cosine_w = {
        'url'   :   1.0,
        'header':   1.0,
        'body'  :   1.0,
        'anchor':   1.0,
        'title' :   1.0     
    }
    bm25f_B = {
        'url'   :   1.0,
        'header':   1.0,
        'body'  :   1.0,
        'anchor':   1.0,
        'title' :   1.0    
    }
    bm25f_W = {
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
        #print >> sys.stderr, self.page.url,self.query.tf_vector,self.page.tf_vector,QueryPage.field_weights       
        self.final_score = DocUtils.cosine_sim(self.query.tf_vector,self.page.tf_vector)
    
    def compute_bm25f_scores(self):
        pass
    
# Look in rank0.main() for how this object is created. Also look at the pa3_play ipython notebook.
class Query(object):

    '''A single query, with all the results associated with it'''
    def __init__(self,query,query_pages,corpus=None):  # query_pages : query -> urls
        self.query = query
        self.terms = self.query.lower().strip().split()
        self.pages = dict([(p,Page(p,v)) for p,v in query_pages.iteritems()]) # URLs
        self.tf_vector = DocUtils.compute_tf_vector(self.terms)
        self.tf_vector = DocUtils.logify(self.tf_vector)
        self.tf_vector = DocUtils.IDFy(self.tf_vector,corpus)
        
    def compute_cosine_scores(self):
        self.page_scores = [QueryPage(self,page) for p,page in self.pages.iteritems()]       
        self.ranked_page_scores = [(qp.page.url,qp.final_score) for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        self.ranked_pages = [qp.page.url for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        return self.ranked_pages        

    def compute_bm25f_scores(self):
        pass
            

class QueryBM25F(object):
    
    '''A single query, with all the results associated with it'''
    def __init__(self,query,query_pages,corpus=None):  # query_pages : query -> urls
        self.query = query
        self.terms = self.query.lower().strip().split()
        self.pages = dict([(p,PageBM25F(p,v)) for p,v in query_pages.iteritems()]) # URLs
        self.raw_tf_map = DocUtils.compute_bm25f_rawtf_map(self.terms, self.pages)
        
        for term,freq in self.raw_tf_map.iteritems():
            for url,freqList in freq.iteritems():
                print term + " - " + url + " - " + str(freqList)
            
        
        #self.tf_vector = Document.compute_tf_vector(self.terms)
        #self.tf_vector = Document.logify(self.tf_vector)
        #self.tf_vector = Document.IDFy(self.tf_vector,corpus)
        pass
        
    def compute_bm25f_scores(self):
        #self.page_scores = [QueryPage(self,page) for p,page in self.pages.iteritems()]       
        #self.ranked_page_scores = [(qp.page.url,qp.final_score) for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        #self.ranked_pages = [qp.page.url for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        
        self.ranked_pages = ["url1","url2","url3"]
        return self.ranked_pages        
