from collections import Counter
import os,sys
from os import path
from math import log
import re
import types
import cPickle as marshal

class SmallestWindowUtils(object):
    '''Container class for SmallestWindow utility static methods'''
    
    @staticmethod
    def get_query_terms_postings_in_field(queryTerms, field):
        query_terms_postings = {}
        
        if isinstance(field,list):
            for term in queryTerms:
                if term in field:
                    query_terms_postings[term] = [p for p in range(len(field)) if term == field[p]]
                else:
                    return {} # the field does not contain all query terms
                
        elif isinstance(field,dict):
            for term in queryTerms:
                if term in field:
                    query_terms_postings[term] = field[term]
                else:
                    return {}

        return query_terms_postings


    @staticmethod
    def get_smallest_window_size(query_terms_posting_list):
        INFINITE = sys.maxsize
        
        query_terms_position = {}
        for qt in query_terms_posting_list:
            query_terms_position[qt] = 0
            
        smallest_window_size = INFINITE - 1
        
        while SmallestWindowUtils.is_query_terms_position_valid(query_terms_position, query_terms_posting_list):
            smallest_window_size = min([smallest_window_size, SmallestWindowUtils.get_window_size(query_terms_position, query_terms_posting_list)])
            query_terms_position = SmallestWindowUtils.update_query_terms_position(query_terms_position, query_terms_posting_list)
            
        return smallest_window_size + 1
    
    @staticmethod    
    def get_window_size(query_terms_position, query_terms_posting_list):
        values = [query_terms_posting_list[qt][query_terms_position[qt]] for qt in query_terms_posting_list]
        return max(values) - min(values)
    
    @staticmethod    
    def is_query_terms_position_valid(query_terms_position, query_terms_posting_list):
        if len(query_terms_posting_list) == 0 or len(query_terms_position) == 0:
            return False
                
        for qt in query_terms_posting_list:
            if query_terms_position[qt] >= len(query_terms_posting_list[qt]):
                return False
        
        return True
    
    @staticmethod    
    def update_query_terms_position(query_terms_position, query_terms_posting_list):
        minvalue = min([query_terms_posting_list[qt][query_terms_position[qt]] for qt in query_terms_posting_list])
        
        for qt in query_terms_posting_list:
            if query_terms_posting_list[qt][query_terms_position[qt]] == minvalue:
                query_terms_position[qt] += 1
                return query_terms_position
            
        return query_terms_position

class DocUtils(object):
    '''Container class for utility static methods'''
    LOGIFY = True
    NORMALIZE = True
    
    @staticmethod
    def calculate_term_raw_frequencies(term, page):
        return dict([(page.url, page.compute_raw_tf_vector(term))])
            
    @staticmethod
    def compute_bm25f_rawtf(queryTerms, page):
        """
        Retuns a map from queryTerm to queryFrequency
        queryFrequency is a map from url to rawFrequencies
        rawFrequencies is a list [term_raw_frequency_url, term_raw_frequency_title, term_raw_frequency_header, term_raw_frequency_body, term_raw_frequency_anchor]
        """
        return dict([ (term, DocUtils.calculate_term_raw_frequencies(term,page)) for term in queryTerms])
    
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
    def features_avg_len(features):
        ''' Returns a list with the avg length of the page features in this format [url,title,header,body,anchor] '''
            
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
                DocUtils.process_body_length(body_length, body_counts, features[query][url])
                DocUtils.process_title_length(title, title_counts, features[query][url])
                DocUtils.process_header_length(header, header_counts, features[query][url])
                DocUtils.process_anchor_length(anchor, anchor_counts, features[query][url])
                DocUtils.process_url_length(url_counts,url)
                
        return [ url_counts[1]    / float(url_counts[0]) if url_counts[0] != 0 else 0.0,
                 title_counts[1]  / float(title_counts[0]) if title_counts[0] != 0 else 0.0,
                 header_counts[1] / float(header_counts[0]) if header_counts[0] != 0 else 0.0,
                 body_counts[1]   / float(body_counts[0]) if body_counts[0] != 0 else 0.0,
                 anchor_counts[1] / float(anchor_counts[0]) if anchor_counts[0] != 0 else 0.0 ]
        
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
        
        self.pagerank         = page_fields.get('pagerank',0)
        self.title            = page_fields.get('title',"")
        self.header           = page_fields.get('header',"")
        self.body_hits        = page_fields.get('body_hits',{})
        self.anchors          = [Anchor(text,count) for text,count in page_fields.get('anchors',{}).iteritems()]
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
    
    fields = ['url', 'title','header','body','anchor']
    
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
        self.pagerank  = page_fields.get('pagerank',0)
                
        # [url,title,header,body,anchors]
        self.fields_length = [ len(self.url_content), len(self.title_content), len(self.headers_content), self.body_length, len(self.anchors_content) ]
        
    def raw_tf_in_url(self, term):
        return float(self.url_content.count(term))
    
    def raw_tf_in_title(self, term):
        return float(self.title_content.count(term))
    
    def raw_tf_in_headers(self, term):
        return float(self.headers_content.count(term))
    
    def raw_tf_in_body(self, term):
        term_postings = self.body_hits.get(term,[])
        return float(len(term_postings))
    
    def raw_tf_in_anchors(self, term):
        return float(self.anchors_content.count(term))

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
        'title' :   1.0,
    }
    
    smallest_window_boost = 2.0
        
    def __init__(self,query,page,useSmallestWindow=False):
        self.query = query
        self.page = page
        self.field_scores = {}
        self.final_score = 0.0
        self.compute_cosine_scores(useSmallestWindow)
        
    def compute_cosine_scores(self, useSmallestWindow=False):
        if useSmallestWindow:
            INFINITE  = sys.maxsize
            QUERY_LEN = len(self.query.terms) 
            B         = 1.0
            
            smallestWindow = self.findSmallestWindow(self.query, self.page)
            
            if smallestWindow == QUERY_LEN:
                B = self.smallest_window_boost
            elif smallestWindow == INFINITE:
                B = 1.0
            elif smallestWindow > QUERY_LEN:
                B_scaled = self.smallest_window_boost * (float(QUERY_LEN) / smallestWindow)
                B = 1.0 if B_scaled < 1.0 else B_scaled
             
            score         = DocUtils.cosine_sim(self.query.tf_vector,self.page.tf_vector)
            adjustedScore = B * score
            
            #print >> sys.stderr, "Query: " + str(self.query.terms) + " Page: " + self.page.url + " Smallest Window: " + str(smallestWindow) + " B: " + str(B) + " score: " + str(score) + " adjustedScore: " + str(adjustedScore)   
            self.final_score = B * score
            
        else:
            #print >> sys.stderr, self.page.url,self.query.tf_vector,self.page.tf_vector,QueryPage.field_weights       
            self.final_score = DocUtils.cosine_sim(self.query.tf_vector,self.page.tf_vector)
            
    def findSmallestWindow(self, query, page):        
        url_terms     = filter(lambda x: len(x) > 0, re.split('\W', page.url))
        title_terms   = page.title.lower().strip().split()
        header_terms  = reduce(lambda x,h: x+h.strip().lower().split(),page.header,[])
        anchors_terms = dict([(anchor.text,anchor.terms) for anchor in page.anchors])
        body_terms    = page.body_hits
        
        query_terms_postings_in_url     = SmallestWindowUtils.get_query_terms_postings_in_field(query.terms, url_terms)
        query_terms_postings_in_title   = SmallestWindowUtils.get_query_terms_postings_in_field(query.terms, title_terms)
        query_terms_postings_in_header  = SmallestWindowUtils.get_query_terms_postings_in_field(query.terms, header_terms)
        query_terms_postings_in_body    = SmallestWindowUtils.get_query_terms_postings_in_field(query.terms, body_terms)

        query_terms_postings_in_anchors = [] # There is an entry in the list for each anchor in the page
        for anchor_terms in anchors_terms.values():
            query_terms_postings_in_anchors.append(SmallestWindowUtils.get_query_terms_postings_in_field(query.terms, anchor_terms))
            
        window_sizes = []
        window_sizes.append(SmallestWindowUtils.get_smallest_window_size(query_terms_postings_in_url))
        window_sizes.append(SmallestWindowUtils.get_smallest_window_size(query_terms_postings_in_title))
        window_sizes.append(SmallestWindowUtils.get_smallest_window_size(query_terms_postings_in_header))
        window_sizes.append(SmallestWindowUtils.get_smallest_window_size(query_terms_postings_in_body))
        
        for query_terms_postings_in_anchor in query_terms_postings_in_anchors:
            window_sizes.append(SmallestWindowUtils.get_smallest_window_size(query_terms_postings_in_anchor))
        
        smallest_window = min(window_sizes)
        #print >> sys.stderr, "Query: " + str(query.terms) + " Page: " + page.url + " Smallest Window: " + str(smallest_window)
        
        return smallest_window

    
class QueryPageBM25F(object):
    
    bm25f_B = [1.0, 1.0, 1.0, 1.0, 1.0]     # [url,title,header,body,anchor]
    bm25f_W = [1.0, 1.0, 1.0, 1.0, 1.0]     # [url,title,header,body,anchor]
    
    K1          = 1.0
    lamd        = 1.0
    
    Vf          = None
    lamd_prime  = 1.0
    lamd_prime2 = 1.0
    
        
    def __init__(self,query,page,fields_avg_len,corpus):
        self.query = query
        self.page = page
        self.field_scores = {}
        self.final_score = 0.0
        self.fields_avg_len = fields_avg_len
        self.corpus = corpus        
        self.compute_bm25f_scores()

            
    def compute_bm25f_scores(self):
        terms_raw_tf_per_field = [ self.page.compute_raw_tf_vector(term) for term in self.query.terms ]  # e.g. [ [1 3 4 1 3], [4 5 2 1 2] ]
        
        # Field dependent normalized tf per term
        fdn_tf = []
        for term_raw_tf in terms_raw_tf_per_field:
            fdn_tf.append( self.compute_fdn_tf(term_raw_tf, QueryPageBM25F.bm25f_B, self.page.fields_length, self.fields_avg_len) )
            
        # Overall weight for each term in this page, across all fields
        weight_tf = [self.compute_weight_tf(term_fdn, QueryPageBM25F.bm25f_W) for term_fdn in fdn_tf]
        
        # Overall score of this page for this query
        self.final_score = self.compute_final_score(self.query.terms, weight_tf, QueryPageBM25F.K1, QueryPageBM25F.lamd, QueryPageBM25F.lamd_prime, QueryPageBM25F.lamd_prime2, QueryPageBM25F.Vf, self.page.pagerank, self.corpus);
        
    
    def compute_fdn_tf(self, raw_tf, B, page_fields_length, fields_avg_len):
        fdn_tf = []
        for idx in xrange(len(raw_tf)):
            B_prime = 1 + B[idx]*(page_fields_length[idx]/fields_avg_len[idx] - 1)
            fdn_tf.append( 0.0 if fields_avg_len[idx]==0.0 or B_prime == 0 else (raw_tf[idx] / B_prime) )
            
        return fdn_tf
    
    def compute_weight_tf(self, fdn_tf, W):
        weight_tf = 0
        for idx in xrange(len(fdn_tf)):
            weight_tf += fdn_tf[idx] * W[idx]
        
        return weight_tf
    
    def compute_final_score(self, terms, weight_tf, K1, lamd, lamd_prime, lamd_prime2, Vf, pagerank, corpus):
        final_score = 0
        for idx in xrange(len(terms)):
            final_score += (weight_tf[idx] * corpus.get_IDF(terms[idx])) / (K1 + weight_tf[idx])
         
        # Non-textual feature (pagerank)   
        final_score += lamd * Vf(self, lamd_prime, pagerank, lamd_prime2)
        
        return final_score
        
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
        
    def compute_cosine_scores(self,useSmallestWindow=False):
        self.page_scores = [QueryPage(self,page,useSmallestWindow) for p,page in self.pages.iteritems()]       
        self.ranked_page_scores = [(qp.page.url,qp.final_score) for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        self.ranked_pages = [qp.page.url for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        return self.ranked_pages        

class QueryBM25F(object):
    
    '''A single query, with all the results associated with it'''
    def __init__(self,query,query_pages,fields_avg_length,corpus):  # query_pages : query -> urls
        self.query = query
        self.terms = self.query.lower().strip().split()
        self.pages = dict([(p,PageBM25F(p,v)) for p,v in query_pages.iteritems()]) # URLs
        self.fields_avg_length = fields_avg_length
        self.corpus = corpus

        
    def compute_bm25f_scores(self):
        self.page_scores = [QueryPageBM25F(self,page,self.fields_avg_length,self.corpus) for p,page in self.pages.iteritems()]       
        self.ranked_page_scores = [(qp.page.url,qp.final_score) for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        self.ranked_pages = [qp.page.url for qp in sorted(self.page_scores,key=lambda x: x.final_score, reverse=True)]
        
        return self.ranked_pages        
