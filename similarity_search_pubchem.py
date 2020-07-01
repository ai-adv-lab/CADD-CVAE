"""
The MIT License

 

Copyright (c) 2020 Samsung SDS

 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""


import chemfp
import chemfp.encodings
import chemfp.bitops
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import Draw
import time
import pandas as pd
import os
from numpy.random import randn

# Required installation : chemfp (python 2.7)
# installed in jupyter virtual environment : source activate chemfp

def search_pubchem(MACCS_bit, thr):
    """
    search
    """
    out=[]
    converted=chemfp.encodings.from_binary_lsb(MACCS_bit)
    for i in range(5323):
        a="./fps/Compound_"+str(i*25000+1).zfill(9) + '_' + str((i+1)*25000).zfill(9) +".fps"
        try:
            arena = chemfp.load_fingerprints(a, reorder=False, format="fps")
            out.extend(chemfp.search.threshold_tanimoto_search_fp(converted[1], arena, thr).get_ids_and_scores())
        except:
            i=i
            #print "No such file or directory: " + a
    return out

def fp_eval_pubchem(filename):
    j=0
    f = open("./"+filename+'.txt', 'r')
    os.mkdir(filename)
    idx=[]
    num_th_80=[]
    num_th_85 =[]
    num_th_90 =[]
    num_th_95 =[]
    num_th_98 =[]


    while True:
        if j==300:
            break
        j += 1
        line = f.readline()
        if not line: break
        now = line.split(",")
        start_time = time.time()
        res = search_pubchem(now[0], thr=0.8)
        print "%d: " % (j)
        print "number of similar compound : %d" % (len(res))
        print "--- %s seconds ---" % (time.time() - start_time)
        idx.append(j)

        cid_array = []
        sim_array = []
        for id in res:
            cid_array += [id[0]]
            sim_array += [id[1]]
        cid_array = [int(i) for i in cid_array]
        result = {'CID': cid_array, 'similarity': sim_array}
        a = pd.DataFrame(result)
        a = a.sort_index(by='similarity', ascending=False)
        a.to_csv(filename+"/result_pubchem_" + str(j) + '.csv', index=False)

        num_th_80.append(a['CID'].describe()[0])

        a_over_85 = a.loc[(a['similarity'] > 0.85)]
        num_th_85.append(a_over_85['CID'].describe()[0])

        a_over_90 = a.loc[(a['similarity'] > 0.90)]
        num_th_90.append(a_over_90['CID'].describe()[0])

        a_over_95 = a.loc[(a['similarity'] > 0.95)]
        num_th_95.append(a_over_95['CID'].describe()[0])

        a_over_98 = a.loc[(a['similarity'] > 0.98)]
        num_th_98.append(a_over_98['CID'].describe()[0])

    eval_result = {'count >=0.8': num_th_80, 'count >=0.85': num_th_85, 'count >=0.9': num_th_90, 'count >=0.95': num_th_95, 'count >=0.98': num_th_98}
    df = pd.DataFrame(eval_result, columns=['count >=0.8','count >=0.85','count >=0.9','count >=0.95','count >=0.98'])
    df.to_csv(filename + '/eval.csv', index=False)
    with open(filename + '/log.txt', 'a') as result_log:
        a8=df.loc[(df['count >=0.8'] > 0)]
        a85 = df.loc[(df['count >=0.85'] > 0)]
        a9 = df.loc[(df['count >=0.9'] > 0)]
        a95 = df.loc[(df['count >=0.95'] > 0)]
        a98 = df.loc[(df['count >=0.98'] > 0)]

        result_log.write("number of fingerprint on DB(>=0.8) : %f\n" % (a8['count >=0.8'].describe()[0]))
        result_log.write("number of fingerprint on DB(>=0.85) : %f\n" % (a85['count >=0.85'].describe()[0]))
        result_log.write("number of fingerprint on DB(>=0.9) : %f\n" % (a9['count >=0.9'].describe()[0]))
        result_log.write("number of fingerprint on DB(>=0.95) : %f\n" % (a95['count >=0.95'].describe()[0]))
        result_log.write("number of fingerprint on DB(>=0.98) : %f\n" % (a98['count >=0.98'].describe()[0]))
    f.close()
epoch = "494"
fp_eval_pubchem("dec_fp_v2_e"+epoch)
fp_eval_pubchem("dec_fp_v4_e"+epoch)
fp_eval_pubchem("dec_fp_v6_e"+epoch)
fp_eval_pubchem("dec_fp_v8_e"+epoch)
fp_eval_pubchem("dec_fp_v10_e"+epoch)
