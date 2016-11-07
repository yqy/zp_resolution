#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy

from conf import *
from buildTree import get_info_from_file
import get_dir
import get_feature
import word2vec
import network

if(len(sys.argv) <= 1): 
    sys.stderr.write("Not specify options, type '-h' for help\n")
    exit()


if args.type == "azp":
    path = args.data

    paths = get_dir.get_all_file(path,[])

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)
        anaphorics = []
        fl = []
        for (sentence_index,zp_index,antecedents,is_azp) in azps:
            if is_azp:
                anaphorics.append((sentence_index,zp_index))
        for (sentence_index,zp_index) in zps:
            nl,wl = nodes_info[sentence_index]
            ifl = get_feature.get_azp_feature_zp(wl[zp_index],wl,[])
            azp_result = "0"
            if (sentence_index,zp_index) in anaphorics:
                azp_result = "1"
            ifl = [azp_result] + ifl 
            fl.append(ifl) 
        get_feature.print_feature(1,fl)


if args.type == "template":
    path = args.data

    paths = get_dir.get_all_file(path,[])
    MAX = 2

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        fl = []
        for (zp_sentence_index,zp_index,antecedents,is_azp) in azps:
            if is_azp:
                for (candi_sentence_index,begin_word_index,end_word_index) in antecedents:
                    anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))

        for (sentence_index,zp_index) in zps:
            zp_nl,zp_wl = nodes_info[sentence_index]
            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    zp = (sentence_index,zp_index)
                    candidate = (candi_sentence_index,candi_begin,candi_end)
                    ifl = get_feature.get_template(zp,candidate,zp_wl,candi_wl,[],[],[])
                    res_result = "0"
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = "1"
                        for item in ifl:
                            print item

if args.type == "res":

    print >> sys.stderr,"read files"
    f = open("./dict/HcPz")
    HcPz = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        HcPz.append(line)
    f.close()

    f = open("./dict/PcPz")
    PcPz = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        PcPz.append(line)
    f.close()

    f = open("./dict/HcP")
    HcP = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        HcP.append(line)
    f.close()


    path = args.data

    paths = get_dir.get_all_file(path,[])
    MAX = 2

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        fl = []
        for (zp_sentence_index,zp_index,antecedents,is_azp) in azps:
            if is_azp:
                for (candi_sentence_index,begin_word_index,end_word_index) in antecedents:
                    anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))

        #for (sentence_index,zp_index) in zps:  ## get feature from all ZPs
        for (sentence_index,zp_index,ax,is_azp) in azps:   ### get feature only from AZPs
            zp_nl,zp_wl = nodes_info[sentence_index]
            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    zp = (sentence_index,zp_index)
                    candidate = (candi_sentence_index,candi_begin,candi_end)
                    #ifl = get_feature.get_res_feature_Max(zp,candidate,zp_wl,candi_wl,[],[],[])
                    #ifl = get_feature.get_res_feature_Max(zp,candidate,zp_wl,candi_wl,HcPz,PcPz,HcP)
                    ifl = get_feature.get_res_feature_Max(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    #ifl = get_feature.get_res_feature_NN(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    res_result = "0"
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = "1"

                    ifl = [res_result] + ifl 
                    fl.append(ifl) 
        get_feature.print_feature(1,fl)


def get_CF_score(res_score,index,score,candi_similarity_matrix):
    similarities = candi_similarity_matrix[index]
    point = 0.0
    norm = 0.0
    for CFi in range(len(score)):
        #if CFi == index:  ## 去掉自己
        #    continue

        similarity = similarities[CFi]
        #similarity = similarities[CFi] * math.pow(abs(similarities[CFi]),1.5)
        this_score = score[CFi]

        point += similarity*this_score
        norm += abs(similarity)

    if norm == 0:
        return res_score
    else:
        return 0.05*point/norm + 0.95*res_score


def get_prf(anaphorics_result,predict_result):
    ## 如果 ZP 是负例 则没有anaphorics_result
    ## 如果 predict 出负例 则 predict_candi_sentence_index = -1
    should = 0
    right = 0
    predict_right = 0
    for i in range(len(predict_result)):
        (sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end) = predict_result[i]
        anaphoric = anaphorics_result[i] 
        if anaphoric:
            should += 1
            if (sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end) in anaphoric:
                right += 1
        if not (predict_candi_sentence_index == -1):
            predict_right += 1

    print "Should:",should,"Right:",right,"PredictRight:",predict_right
    if predict_right == 0:
        P = 0.0
    else:
        P = float(right)/float(predict_right)

    if should == 0:
        R = 0.0
    else:
        R = float(right)/float(should)

    if (R == 0.0) or (P == 0.0):
        F = 0.0
    else:
        F = 2.0/(1.0/P + 1.0/R)

    print "P:",P
    print "R:",R
    print "F:",F


def get_sentence(zp_sentence_index,zp_index,nodes_info):
    #返回只包含zp_index位置的ZP的句子
    nl,wl = nodes_info[zp_sentence_index]
    return_words = []
    for i in range(len(wl)):
        this_word = wl[i].word
        if i == zp_index:
            return_words.append("**pro**")
        else:
            if not (this_word == "*pro*"): 
                return_words.append(this_word)
    return " ".join(return_words)

def get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result):
    nl,wl = nodes_info[candi_sentence_index]
    candi_word = []
    for i in range(candi_begin,candi_end+1):
        candi_word.append(wl[i].word)
    candi_word = "_".join(candi_word)

    candi_info = [str(res_result),candi_word]
    return candi_info
if args.type == "gold":

    print >> sys.stderr,"read files"
    f = open("./dict/HcPz")
    HcPz = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        HcPz.append(line)
    f.close()

    f = open("./dict/PcPz")
    PcPz = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        PcPz.append(line)
    f.close()

    f = open("./dict/HcP")
    HcP = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        HcP.append(line)
    f.close()

    print >> sys.stderr, "Read W2V"
    w2v = word2vec.Word2Vec(args.embedding)


    FeatureFile = "./tmp_data/feature."
    path = args.data
    paths = get_dir.get_all_file(path,[])
    MAX = 2
    hits = 0

    predict_result = []
    CF_result = []
    anaphorics_result = []
    
    done_zp_num = 0

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,is_azp) in azps:
            if is_azp:
                for (candi_sentence_index,begin_word_index,end_word_index) in antecedents:
                    anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                    ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue
            
            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence

            fl = []
            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_for_test = []
            candi_embeddings = []
             
            candi_info_for_log = []
            candi_number = 0

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    zp = (sentence_index,zp_index)
                    candidate = (candi_sentence_index,candi_begin,candi_end)
                    candi_for_test.append(candidate) 

                    #ifl = get_feature.get_res_feature_Max(zp,candidate,zp_wl,candi_wl,[],[],[])
                    ifl = get_feature.get_res_feature_Max(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    res_result = "0"
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = "1"

                    ifl = [res_result] + ifl 
                    fl.append(ifl) 
                    
                    ### get embedding for each NP candidates ###
                    candi_word = []
                    for i in range (candi_begin,candi_end+1):
                        if candi_wl[i].word.find("*") < 0:
                            candi_word.append(candi_wl[i].word)
                    candi_embeddings.append(w2v.get_vector_by_list(candi_word))
           
                    ### candi_info : [index,res_result,candi_words]
                    candi_info = get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result)
                    candi_info = [str(candi_number)] + candi_info
                    candi_info_for_log.append(candi_info)
                    
                    candi_number += 1
 
            ### calculate similarity between candidates ###
            candi_similarity_matrix = [] ### similarity matrix
            for i in range(len(candi_embeddings)):
                simis = []
                for j in range(len(candi_embeddings)):
                    simis.append(word2vec.cosin(candi_embeddings[i],candi_embeddings[j]))
                candi_similarity_matrix.append(simis)


            ###  write feature and get result ! ###
            get_feature.write_feature_file_MaxEnt(FeatureFile+"res.trick",fl,sentence_index)
            cmd = "./start_maxEnt.sh res.trick > ./tmp_data/t"
            os.system(cmd)
            result_file = "./tmp_data/result.res.trick"
            class_result,score = get_feature.read_result_Max_with_index(result_file,float(args.res_t),args.res_pos)

            ### nearest first search strategy ###
            threshold = float(args.res_t)
            #threshold = 0.0
            ### best first search strategy ###
            CF_threshold = 0.0
            num = 0

            predict_candi_sentence_index = -1
            predict_candi_begin = -1
            predict_candi_end = -1

            CF_predict_candi_sentence_index = -1
            CF_predict_candi_begin = -1
            CF_predict_candi_end = -1

            ## for log 记录选了第一个candidate
            CF_candi_index = -1
            predict_candi_index = -1

            for i in range(len(candi_for_test)):
                candi_sentence_index,candi_begin,candi_end = candi_for_test[i]
                res_score = score[i]

                CF_score = get_CF_score(res_score,i,score,candi_similarity_matrix)

                candi_info_for_log[i].append("Res_score:%f"%res_score)
                candi_info_for_log[i].append("CF_score:%f"%CF_score)

                ### for CF ###
                if CF_score >= CF_threshold:
                    CF_predict_candi_sentence_index = candi_sentence_index
                    CF_predict_candi_begin = candi_begin 
                    CF_predict_candi_end = candi_end
                    CF_threshold = CF_score
                    CF_candi_index = i
    
                ### for normal resolution score ###
                if res_score >= threshold:
                    predict_candi_sentence_index = candi_sentence_index
                    predict_candi_begin = candi_begin 
                    predict_candi_end = candi_end
                    predict_candi_index = i

            if predict_candi_sentence_index == -1 and len(candi_for_test) > 0:
                candi_sentence_index,candi_begin,candi_end = candi_for_test[-1]
                predict_candi_sentence_index = candi_sentence_index
                predict_candi_begin = candi_begin 
                predict_candi_end = candi_end
                predict_candi_index = len(candi_for_test)-1

            if CF_predict_candi_sentence_index == -1 and len(candi_for_test) > 0:
                candi_sentence_index,candi_begin,candi_end = candi_for_test[-1]
                CF_predict_candi_sentence_index = candi_sentence_index
                CF_predict_candi_begin = candi_begin 
                CF_predict_candi_end = candi_end
                CF_candi_index = len(candi_for_test)-1

            print >> sys.stderr, "Candidates:"
            for candi_info in candi_info_for_log:
                print >> sys.stderr,"\t".join(candi_info)
            print >> sys.stderr, "Results:" 
            if predict_candi_index >= 0:
                print >> sys.stderr, "Predict -- ","\t".join(candi_info_for_log[predict_candi_index])
            if CF_candi_index >= 0:
                print >> sys.stderr, "CF -- ","\t".join(candi_info_for_log[CF_candi_index])
            print >> sys.stderr, "Done ZP #%d"%(done_zp_num) 
            predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end)) 
            CF_result.append((sentence_index,zp_index,CF_predict_candi_sentence_index,CF_predict_candi_begin,CF_predict_candi_end)) 
            anaphorics_result.append(anaphorics)
    
    get_prf(anaphorics_result,predict_result)
    get_prf(anaphorics_result,CF_result)


if args.type == "auto":

    print >> sys.stderr,"read files"
    f = open("./dict/HcPz")
    HcPz = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        HcPz.append(line)
    f.close()

    f = open("./dict/PcPz")
    PcPz = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        PcPz.append(line)
    f.close()

    f = open("./dict/HcP")
    HcP = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        HcP.append(line)
    f.close()

    print >> sys.stderr, "Read W2V"
    w2v = word2vec.Word2Vec(args.embedding)


    FeatureFile = "./tmp_data/feature."
    path = args.data
    paths = get_dir.get_all_file(path,[])
    MAX = 2
    hits = 0

    predict_result = []
    CF_result = []
    anaphorics_result = []
    
    done_zp_num = 0

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,is_azp) in azps:
            if is_azp:
                for (candi_sentence_index,begin_word_index,end_word_index) in antecedents:
                    anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                    ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:
            done_zp_num += 1
            
            fl = []
            ifl = []
            zp_nl,zp_wl = nodes_info[sentence_index]
            ifl = get_feature.get_azp_feature_zp(zp_wl[zp_index],zp_wl,[])
            azp_result = "NOT" 
            if (sentence_index,zp_index) in ana_zps:
                azp_result = "IS" 
            ifl = ["0"] + ifl 
            fl.append(ifl)
            
            get_feature.write_feature_file_MaxEnt(FeatureFile+"azp",fl,sentence_index)
            cmd = "./start_maxEnt.sh azp > ./tmp_data/t"
            os.system(cmd)
            result_file = "./tmp_data/result.azp"
            class_result,score = get_feature.read_result_Max_with_index(result_file,float(args.azp_t),args.res_pos)

            azp_score = score[0] ##每次只有一个实例
            if azp_score < float(args.azp_t):
                print >> sys.stderr,"------" 
                this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
                print >> sys.stderr, "-Sentence:",this_sentence
                print >> sys.stderr, "Predict Not an AZP"
                print >> sys.stderr, "Real: %s an AZP"%(azp_result)
                print >> sys.stderr, "Done ZP #%d"%(done_zp_num) 
                predict_result.append((-1,-1,-1,-1,-1)) 
                CF_result.append((-1,-1,-1,-1,-1)) 
                if azp_result == "IS":
                    anaphorics_result.append(anaphorics)
                else:
                    anaphorics_result.append(None)
                continue
            #if not (sentence_index,zp_index) in ana_zps:
            #    continue
            
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence

            fl = []
            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_for_test = []
            candi_embeddings = []
             
            candi_info_for_log = []
            candi_number = 0

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    zp = (sentence_index,zp_index)
                    candidate = (candi_sentence_index,candi_begin,candi_end)
                    candi_for_test.append(candidate) 

                    #ifl = get_feature.get_res_feature_Max(zp,candidate,zp_wl,candi_wl,[],[],[])
                    ifl = get_feature.get_res_feature_Max(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    res_result = "0"
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = "1"

                    ifl = [res_result] + ifl 
                    fl.append(ifl) 
                    
                    ### get embedding for each NP candidates ###
                    candi_word = []
                    for i in range (candi_begin,candi_end+1):
                        if candi_wl[i].word.find("*") < 0:
                            candi_word.append(candi_wl[i].word)
                    candi_embeddings.append(w2v.get_vector_by_list(candi_word))
           
                    ### candi_info : [index,res_result,candi_words]
                    candi_info = get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result)
                    candi_info = [str(candi_number)] + candi_info
                    candi_info_for_log.append(candi_info)
                    
                    candi_number += 1
 
            ### calculate similarity between candidates ###
            candi_similarity_matrix = [] ### similarity matrix
            for i in range(len(candi_embeddings)):
                simis = []
                for j in range(len(candi_embeddings)):
                    simis.append(word2vec.cosin(candi_embeddings[i],candi_embeddings[j]))
                candi_similarity_matrix.append(simis)


            ###  write feature and get result ! ###
            get_feature.write_feature_file_MaxEnt(FeatureFile+"res.trick",fl,sentence_index)
            cmd = "./start_maxEnt.sh res.trick > ./tmp_data/t"
            os.system(cmd)
            result_file = "./tmp_data/result.res.trick"
            class_result,score = get_feature.read_result_Max_with_index(result_file,float(args.res_t),args.res_pos)

            ### nearest first search strategy ###
            threshold = float(args.res_t)
            #threshold = 0.0
            ### best first search strategy ###
            CF_threshold = 0.0
            num = 0

            predict_candi_sentence_index = -1
            predict_candi_begin = -1
            predict_candi_end = -1

            CF_predict_candi_sentence_index = -1
            CF_predict_candi_begin = -1
            CF_predict_candi_end = -1

            ## for log 记录选了第一个candidate
            CF_candi_index = -1
            predict_candi_index = -1

            for i in range(len(candi_for_test)):
                candi_sentence_index,candi_begin,candi_end = candi_for_test[i]
                res_score = score[i]

                CF_score = get_CF_score(res_score,i,score,candi_similarity_matrix)

                candi_info_for_log[i].append("Res_score:%f"%res_score)
                candi_info_for_log[i].append("CF_score:%f"%CF_score)

                ### for CF ###
                if CF_score >= CF_threshold:
                    CF_predict_candi_sentence_index = candi_sentence_index
                    CF_predict_candi_begin = candi_begin 
                    CF_predict_candi_end = candi_end
                    CF_threshold = CF_score
                    CF_candi_index = i
    
                ### for normal resolution score ###
                if res_score >= threshold:
                    predict_candi_sentence_index = candi_sentence_index
                    predict_candi_begin = candi_begin 
                    predict_candi_end = candi_end
                    predict_candi_index = i

            if predict_candi_sentence_index == -1 and len(candi_for_test) > 0:
                candi_sentence_index,candi_begin,candi_end = candi_for_test[-1]
                predict_candi_sentence_index = candi_sentence_index
                predict_candi_begin = candi_begin 
                predict_candi_end = candi_end
                predict_candi_index = len(candi_for_test)-1

            if CF_predict_candi_sentence_index == -1 and len(candi_for_test) > 0:
                candi_sentence_index,candi_begin,candi_end = candi_for_test[-1]
                CF_predict_candi_sentence_index = candi_sentence_index
                CF_predict_candi_begin = candi_begin 
                CF_predict_candi_end = candi_end
                CF_candi_index = len(candi_for_test)-1

            print >> sys.stderr, "Candidates:"
            for candi_info in candi_info_for_log:
                print >> sys.stderr,"\t".join(candi_info)
            print >> sys.stderr, "Results: %s AZP"%(azp_result) 
            if predict_candi_index >= 0:
                print >> sys.stderr, "Predict -- ","\t".join(candi_info_for_log[predict_candi_index])
            if CF_candi_index >= 0:
                print >> sys.stderr, "CF -- ","\t".join(candi_info_for_log[CF_candi_index])
            print >> sys.stderr, "Done ZP #%d"%(done_zp_num) 
            predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end)) 
            CF_result.append((sentence_index,zp_index,CF_predict_candi_sentence_index,CF_predict_candi_begin,CF_predict_candi_end)) 
            #anaphorics_result.append(anaphorics)
            if azp_result == "IS":
                anaphorics_result.append(anaphorics)
            else:
                anaphorics_result.append(None)
 
    print "Predict:"
    get_prf(anaphorics_result,predict_result)
    print "CF:"
    get_prf(anaphorics_result,CF_result)


from subprocess import *
if args.type == "system":

    print >> sys.stderr,"read files"
    f = open("./dict/HcPz")
    HcPz = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        HcPz.append(line)
    f.close()

    f = open("./dict/PcPz")
    PcPz = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        PcPz.append(line)
    f.close()

    f = open("./dict/HcP")
    HcP = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        HcP.append(line)
    f.close()

    parser = Popen(["./go_parse.sh"] ,stdout=PIPE,stdin=PIPE,shell = True)
    parser_in = parser.stdin
    parser_out = parser.stdout
    parser_in.write("1\n")
    tmp = parser_out.readline() 

    print >> sys.stderr, "Read W2V"
    w2v = word2vec.Word2Vec(args.embedding)


    FeatureFile = "./tmp_data/feature."
    path = args.data
    paths = get_dir.get_all_file(path,[])
    MAX = 2
    hits = 0

    predict_result = []
    CF_result = []
    anaphorics_result = []
    
    done_zp_num = 0

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file_system(file_name,parser_in,parser_out,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,is_azp) in azps:
            if is_azp:
                for (candi_sentence_index,begin_word_index,end_word_index) in antecedents:
                    anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                    ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:
            done_zp_num += 1

            fl = []
            ifl = []
            zp_nl,zp_wl = nodes_info[sentence_index]
            ifl = get_feature.get_azp_feature_zp(zp_wl[zp_index],zp_wl,[])
            azp_result = "NOT" 
            if (sentence_index,zp_index) in ana_zps:
                azp_result = "IS" 
            ifl = ["0"] + ifl 
            fl.append(ifl)
            
            get_feature.write_feature_file_MaxEnt(FeatureFile+"azp",fl,sentence_index)
            cmd = "./start_maxEnt.sh azp > ./tmp_data/t"
            os.system(cmd)
            result_file = "./tmp_data/result.azp"
            class_result,score = get_feature.read_result_Max_with_index(result_file,float(args.azp_t),args.res_pos)

            azp_score = score[0] ##每次只有一个实例
            if azp_score < float(args.azp_t):
                print >> sys.stderr,"------" 
                this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
                print >> sys.stderr, "-Sentence:",this_sentence
                print >> sys.stderr, "Predict Not an AZP"
                print >> sys.stderr, "Real: %s an AZP"%(azp_result)
                print >> sys.stderr, "Done ZP #%d"%(done_zp_num) 
                predict_result.append((-1,-1,-1,-1,-1)) 
                CF_result.append((-1,-1,-1,-1,-1)) 
                if azp_result == "IS":
                    anaphorics_result.append(anaphorics)
                else:
                    anaphorics_result.append(None)
                continue
            #if not (sentence_index,zp_index) in ana_zps:
            #    continue
            
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence

            fl = []
            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_for_test = []
            candi_embeddings = []
             
            candi_info_for_log = []
            candi_number = 0

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    zp = (sentence_index,zp_index)
                    candidate = (candi_sentence_index,candi_begin,candi_end)
                    candi_for_test.append(candidate) 

                    #ifl = get_feature.get_res_feature_Max(zp,candidate,zp_wl,candi_wl,[],[],[])
                    ifl = get_feature.get_res_feature_Max(zp,candidate,zp_wl,candi_wl,[],[],HcP)
                    res_result = "0"
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = "1"

                    ifl = [res_result] + ifl 
                    fl.append(ifl) 
                    
                    ### get embedding for each NP candidates ###
                    candi_word = []
                    for i in range (candi_begin,candi_end+1):
                        if candi_wl[i].word.find("*") < 0:
                            candi_word.append(candi_wl[i].word)
                    candi_embeddings.append(w2v.get_vector_by_list(candi_word))
           
                    ### candi_info : [index,res_result,candi_words]
                    candi_info = get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result)
                    candi_info = [str(candi_number)] + candi_info
                    candi_info_for_log.append(candi_info)
                    
                    candi_number += 1
 
            ### calculate similarity between candidates ###
            candi_similarity_matrix = [] ### similarity matrix
            for i in range(len(candi_embeddings)):
                simis = []
                for j in range(len(candi_embeddings)):
                    simis.append(word2vec.cosin(candi_embeddings[i],candi_embeddings[j]))
                candi_similarity_matrix.append(simis)


            ###  write feature and get result ! ###
            get_feature.write_feature_file_MaxEnt(FeatureFile+"res.trick",fl,sentence_index)
            cmd = "./start_maxEnt.sh res.trick > ./tmp_data/t"
            os.system(cmd)
            result_file = "./tmp_data/result.res.trick"
            class_result,score = get_feature.read_result_Max_with_index(result_file,float(args.res_t),args.res_pos)

            ### nearest first search strategy ###
            threshold = float(args.res_t)
            #threshold = 0.0
            ### best first search strategy ###
            CF_threshold = 0.0
            num = 0

            predict_candi_sentence_index = -1
            predict_candi_begin = -1
            predict_candi_end = -1

            CF_predict_candi_sentence_index = -1
            CF_predict_candi_begin = -1
            CF_predict_candi_end = -1

            ## for log 记录选了第一个candidate
            CF_candi_index = -1
            predict_candi_index = -1

            for i in range(len(candi_for_test)):
                candi_sentence_index,candi_begin,candi_end = candi_for_test[i]
                res_score = score[i]

                CF_score = get_CF_score(res_score,i,score,candi_similarity_matrix)

                candi_info_for_log[i].append("Res_score:%f"%res_score)
                candi_info_for_log[i].append("CF_score:%f"%CF_score)

                ### for CF ###
                if CF_score >= CF_threshold:
                    CF_predict_candi_sentence_index = candi_sentence_index
                    CF_predict_candi_begin = candi_begin 
                    CF_predict_candi_end = candi_end
                    CF_threshold = CF_score
                    CF_candi_index = i
    
                ### for normal resolution score ###
                if res_score >= threshold:
                    predict_candi_sentence_index = candi_sentence_index
                    predict_candi_begin = candi_begin 
                    predict_candi_end = candi_end
                    predict_candi_index = i

            if predict_candi_sentence_index == -1 and len(candi_for_test) > 0:
                candi_sentence_index,candi_begin,candi_end = candi_for_test[-1]
                predict_candi_sentence_index = candi_sentence_index
                predict_candi_begin = candi_begin 
                predict_candi_end = candi_end
                predict_candi_index = len(candi_for_test)-1

            if CF_predict_candi_sentence_index == -1 and len(candi_for_test) > 0:
                candi_sentence_index,candi_begin,candi_end = candi_for_test[-1]
                CF_predict_candi_sentence_index = candi_sentence_index
                CF_predict_candi_begin = candi_begin 
                CF_predict_candi_end = candi_end
                CF_candi_index = len(candi_for_test)-1

            print >> sys.stderr, "Candidates:"
            for candi_info in candi_info_for_log:
                print >> sys.stderr,"\t".join(candi_info)
            print >> sys.stderr, "Results: %s AZP"%(azp_result) 
            if predict_candi_index >= 0:
                print >> sys.stderr, "Predict -- ","\t".join(candi_info_for_log[predict_candi_index])
            if CF_candi_index >= 0:
                print >> sys.stderr, "CF -- ","\t".join(candi_info_for_log[CF_candi_index])
            print >> sys.stderr, "Done ZP #%d"%(done_zp_num) 
            predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end)) 
            CF_result.append((sentence_index,zp_index,CF_predict_candi_sentence_index,CF_predict_candi_begin,CF_predict_candi_end)) 
            #anaphorics_result.append(anaphorics)
            if azp_result == "IS":
                anaphorics_result.append(anaphorics)
            else:
                anaphorics_result.append(None)

    print "Predict:"
    get_prf(anaphorics_result,predict_result)
    print "CF:"
    get_prf(anaphorics_result,CF_result)



def get_inputs(w2v,nodes_info,sentence_index,begin_index,end_index,ty):
    if ty == "zp":
        ### get for ZP ###
        tnl,twl = nodes_info[sentence_index]

        pre_zp_x = []
        pre_zp_x.append(list([0.0]*args.embedding_dimention))
        for i in range(0,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    pre_zp_x.append(list(em_x))

        post_zp_x = []
        for i in range(end_index+1,len(twl)):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    post_zp_x.append(list(em_x))
        post_zp_x.append(list([0.0]*args.embedding_dimention))
        post_zp_x = post_zp_x[::-1]
        return (numpy.array(pre_zp_x),numpy.array(post_zp_x))

    elif ty == "np":
        tnl,twl = nodes_info[sentence_index]
        np_x = []
        np_x.append(list([0.0]*args.embedding_dimention))

        for i in range(begin_index-1,begin_index):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x.append(list(em_x))

        for i in range(begin_index,end_index+1):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x.append(list(em_x))

        for i in range(end_index+1,end_index+2):
            if i >= 0 and i < len(twl):
                em_x = w2v.get_vector_by_word_dl(twl[i].word)
                if em_x is not None:
                    np_x.append(list(em_x))
        return numpy.array(np_x)

if args.type == "nn":

    print >> sys.stderr, "Read W2V"
    w2v = word2vec.Word2Vec(args.embedding)


    path = args.data
    paths = get_dir.get_all_file(path,[])
    MAX = 2
    hits = 0

    training_instances = []
    
    done_zp_num = 0
    
    ####  Training process  ####

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,is_azp) in azps:
            if is_azp:
                for (candi_sentence_index,begin_word_index,end_word_index) in antecedents:
                    anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                    ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue
            
            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence

            fl = []
            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    zp = (sentence_index,zp_index)
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp")
                    np_x = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    training_instances.append((zp_x_pre,zp_x_post,np_x,res_result))
    #for zp_x_pre,zp_x_post,np_x,res_result in training_instances:
    #    print zp_x_pre,zp_x_post,np_x,res_result



    ####  Test process  ####

    path = args.test_data
    paths = get_dir.get_all_file(path,[])
    test_instances = []
    anaphorics_result = []
    
    done_zp_num = 0

    for file_name in paths:
        file_name = file_name.strip()
        print >> sys.stderr, "Read File:%s <<-->> %d/%d"%(file_name,paths.index(file_name)+1,len(paths))

        zps,azps,candi,nodes_info = get_info_from_file(file_name,2)

        anaphorics = []
        ana_zps = []
        for (zp_sentence_index,zp_index,antecedents,is_azp) in azps:
            if is_azp:
                for (candi_sentence_index,begin_word_index,end_word_index) in antecedents:
                    anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                    ana_zps.append((zp_sentence_index,zp_index))

        for (sentence_index,zp_index) in zps:

            if not (sentence_index,zp_index) in ana_zps:
                continue

            this_zp_test_instence = []            
            done_zp_num += 1
   
            print >> sys.stderr,"------" 
            this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
            print >> sys.stderr, "Sentence:",this_sentence

            fl = []
            zp_nl,zp_wl = nodes_info[sentence_index]
            candi_number = 0

            for ci in range(max(0,sentence_index-MAX),sentence_index+1):
                
                candi_sentence_index = ci
                candi_nl,candi_wl = nodes_info[candi_sentence_index] 

                for (candi_begin,candi_end) in candi[candi_sentence_index]:
                    if ci == sentence_index and candi_end > zp_index:
                        continue
                    zp = (sentence_index,zp_index)
                    candidate = (candi_sentence_index,candi_begin,candi_end)

                    zp_x_pre,zp_x_post = get_inputs(w2v,nodes_info,sentence_index,zp_index,zp_index,"zp")
                    np_x = get_inputs(w2v,nodes_info,candi_sentence_index,candi_begin,candi_end,"np")

                    res_result = 0
                    if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                        res_result = 1

                    this_nodes_info = {} ## 为了节省存储空间
                    this_nodes_info[candi_sentence_index] = nodes_info[candi_sentence_index]
                    this_nodes_info[sentence_index] = nodes_info[sentence_index]

                    #this_zp_test_instence.append((zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,nodes_info))
                    this_zp_test_instence.append((zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,this_nodes_info))
            
            anaphorics_result.append(anaphorics)
            test_instances.append(this_zp_test_instence)
    #for test_instance in test_instances:
    #    for zp_x_pre,zp_x_post,np_x,res_result,zp,candidate in test_instance:
    #        print zp_x_pre,zp_x_post,np_x,res_result


    w2v = None # 释放空间

    ##### begin train and test #####
    
    ## Build DL Model ## 
    print >> sys.stderr,"Building Model ..."
    LSTM = network.NetWork(args.embedding_dimention,100,[200,2])
    
    ### 均衡正负例 ###
    pos_num = 0
    neg_num = 0
    training = []   # 存储正负例均衡后的数据
    for zp_x_pre,zp_x_post,np_x,res_result in training_instances:
        if res_result == 1: #正例 
            pos_num += 1
            training.append((zp_x_pre,zp_x_post,np_x,res_result)) 
        else:
            if neg_num <= pos_num:
                training.append((zp_x_pre,zp_x_post,np_x,res_result)) 
                neg_num += 1
    training_instances = [] # 释放空间
    
    for echo in range(args.echos): 
        print >> sys.stderr, "Echo for time",echo
        start_time = timeit.default_timer()
        for zp_x_pre,zp_x_post,np_x,res_result in training:
            LSTM.train_step(np_x,zp_x_pre,zp_x_post,res_result,args.lr)
        end_time = timeit.default_timer()
        print >> sys.stderr,"Parameters"
        LSTM.show_para()
        print >> sys.stderr, end_time - start_time, "seconds!"
    
        #### Test for each echo ####
        
        #predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
        print >> sys.stderr, "Begin test" 
        predict_result = []
        numOfZP = 0
        for test_instance in test_instances:
            numOfZP += 1
            if len(test_instance) == 0: ## no suitable candidates
                predict_result.append((-1,-1,-1,-1,-1))
            else:
                
                zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,nodes_info = test_instance[-1]
                sentence_index,zp_index = zp
                print >> sys.stderr,"------" 
                this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
                print >> sys.stderr, "Sentence:",this_sentence

                print >> sys.stderr, "Candidates:"

                st_score = 0.0
                predict_items = None
                numOfCandi = 0
                predict_str_log = None
                last = ""
                for zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,nodes_info in test_instance:
                    nn_predict = LSTM.get_out(np_x,zp_x_pre,zp_x_post)[0][1]
                    candi_sentence_index,candi_begin,candi_end = candidate
                    candi_str = "\t".join(get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result))
                    if nn_predict >= st_score: 
                        predict_items = (zp,candidate)
                        st_score = nn_predict
                        predict_str_log = "%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    print >> sys.stderr,"%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    last = "%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    numOfCandi += 1

                if predict_items is None:
                    zp_x_pre,zp_x_post,np_x,res_result,zp,candidate,nodes_info = test_instance[-1]
                    predict_items = (zp,candidate)
                    predict_str_log = last

                predict_zp,predict_candidate = predict_items
                sentence_index,zp_index = predict_zp 
                predict_candi_sentence_index,predict_candi_begin,predict_candi_end = predict_candidate

                predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
                print >> sys.stderr, "Results:"
                print >> sys.stderr, "Predict -- %s"%predict_str_log
                print >> sys.stderr, "Done ZP #%d/%d"%(numOfZP,len(test_instances))
        print "Echo",echo 
        get_prf(anaphorics_result,predict_result)
        sys.stdout.flush()

