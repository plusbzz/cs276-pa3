## CS276: Programming Assignment 3

## Instructions
Copy the data/ folder over from PA1 to this directory before running anything

New best: 0.868612756726 {'url': -1705, 'header': -85, 'body': -97, 'anchor': -17, 'title': -288}



Data for the report 
==================================

* Summary

-------------
TASK2 - BM25F
-------------
function = Logarithmic
Sigmoid gave us a slightly better NDCG i.e. 0.8564 vs 0.8562 (can we mention NDCG values in the report?)
However we chose Logarithmic because we think it maybe more stable across test sets (feel free to change this justification) )

# [url,title,header,body,anchor]
QueryPageBM25F.bm25f_B     = [1.0,0.1,1.0,1.0,0.1]
QueryPageBM25F.bm25f_W     = [1.0,0.9,0.8,0.9,0.7]
QueryPageBM25F.K1          = 1
QueryPageBM25F.lamd        = 3.0
QueryPageBM25F.lamd_prime  = 2.0
QueryPageBM25F.lamd_prime2 = 1.0

----------------------------
FOR TASK3 - SMALLEST WINDOW
----------------------------
function = Inverse
B        = 16


    
* Details


Q) For a function that includes the smallest window as one component, how does varying B and the boost function change the performance of the algorithm

For Inverse Function varying the B value allowed us to get a max improvement of 0.52%. The improvement we observed ranged from +0.06% for B=1024 to +0.52%
for B=16. After reaching B=16 we didn't get any better NDCG value by increasing B. This may be to the fact that documents where getting a high boost based on fields that were not
very relevant.

For Sigmoid Function varying the B value allowed us to get a max improvement of 0.23%. The improvement we observed ranged from +0.07% for B=2 to B=0.23% for B=16. After reaching B=16 we didn't get any better NDCG value by increasing B. This may be to the fact that documents where getting a high boost based on fields that were not
very relevant.

url    = -100,
title  = -90,
body   = -90,
header = -80,
anchor = -10

NDCG (ranking_cosine) = 0.8343

B = (2,4,8,16,32,64,128,256,512,1024)

With Inverse Function
NDCG (ranking_cosine_smallest_window per B) = (0.8376,0.8380,0.8374,0.8387,0.8372,0.8363,0.8354,0.8349,0.8349,0.8348)

With Sigmoid Function
NDCG (ranking_cosine_smallest_window per B) = (0.8349,0.8329,0.8350,0.8363,0.8359,0.8356,0.8351,0.8357,0.8355,0.8355)


Q) BM25F, how Bs lambdas and K1 parameters affect the ranking functions

Logarithmic:

    # [url,title,header,body,anchor]
    QueryPageBM25F.bm25f_B     = [1.0,0.1,1.0,1.0,0.1]
    QueryPageBM25F.bm25f_W     = [1.0,0.9,0.8,0.9,0.7]
    QueryPageBM25F.K1    = 1
    QueryPageBM25F.lamd        = 3.0
    QueryPageBM25F.lamd_prime  = 2.0
    QueryPageBM25F.lamd_prime2 = 1.0
    NDCG = 0.8562
    
sigmoid:

    # [url,title,header,body,anchor]
    QueryPageBM25F.bm25f_B     = [1.0,0.5,0.5,0.5,0.1]
    QueryPageBM25F.bm25f_W     = [1.0,0.9,0.8,0.9,0.7]
    QueryPageBM25F.K1    = 1.2
    QueryPageBM25F.lamd        = 25.0
    QueryPageBM25F.lamd_prime  = 1.0   
    QueryPageBM25F.lamd_prime2 = 0.5
    NDCG = 0.8542
    
saturation:
    QueryPageBM25F.bm25f_B     = [1.0,1.0,0.5,0.5,0.1]
    QueryPageBM25F.bm25f_W     = [1.0,0.9,0.8,0.9,0.7]
    QueryPageBM25F.K1          = 1
    QueryPageBM25F.lamd        = 25.0
    QueryPageBM25F.lamd_prime  = 20.0   
    QueryPageBM25F.lamd_prime2 = 1.0
    NDCG = 0.8564
==================================


## Instructions from Staff
This folder contains the following files:

1. Data
  a. queryDocTrainData
     This file contains the training data for this assignment. For each (query,url) pair, there are several features given (details available in the assignment description)
  b. queryDocTrainRel
     This file contains the relevance values for each (query,url) pair given in the queryDocTrainData file. This file can be used for evaluation while building the model
  c. AllQueryTerms
     This file contains the tokens contained in query terms "across train and test data"

2. Helper code 
  a. rank0.py
     This is a baseline skeleton code provided for your help. It contains functions to parse the features data and write the ranked results to stdout. You may or may not use this code, just make sure your output format is the same as the one produced by this file(and mentioned in the handout). 

     The baseline simply ranks the urls in decreasing order of number of body_hits across all query terms.
  b. ndcg.py
     This is the code for calculating the ndcg score of your ranking algorithm. You can run the code as follows:
       $ python ndcg.py <your ranked file> <file with relevance values>

     For example, if you store the results of baseline in a file called "ranked.text", in order to calculate it's ndcg score, you can run the following command:
       $ python ndcg.py ranked.txt queryDocTrainRel

3. rank.sh
   This is the script we will be calling to execute your program. The script takes 2 arguments: 1) the id of the task (0/1/2/3/4, 4 is for extra credit, 0 for baseline), 2) input data file (in the specified format). Therefore, in order to run the baseline code, you can execute:
       $ ./rank.sh 0 queryDocTrainData

   You can use any language to do the assignment as long as you follow two requirements:
     - rank.sh should work with the two parameters as mentioned above
     - rank.sh should output your ranking results in the correct format to stdout
     - your code can take any number of extra arguments, the script should only take these two
     - the way the script is written right now, it assumes that the files for the tasks are called rank1.py, rank2.py, rank3.py, rank4.py (extra credit). You can change the script if you want as long as it meets the input/output requirements

4. submit.py
   This is the submit script used for the assignment. Please submit each task (and report) individually. In order to submit a task, simply run the following command:
       $ python submit.py

   and follow the instructions. Note that 1/2/3 are tasks mentioned in the assignment, 0 is for the report and 4 is for extra credit (optional). The report should be present in the same folder with the name "report.pdf"
