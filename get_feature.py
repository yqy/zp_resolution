#coding=utf8
import sys
import os

def get_same(a,b):
    times = 0 
    for i in range(min(len(a),len(b))):
        if a[i] == b[i]:
            times += 1
        else:
            break    
    return times


def toString(node_list):
    #将node_list里的word连成字符串
    out = []
    for node in node_list:
        out.append(node.word)
    return "_".join(out)

def read_result_Max_with_index(filename,t,target):
    result = []
    score = []
    f = open(filename)
    ii = 0
    while True:
        line = f.readline()
        if not line:break

        ii += 1

        line = line.strip().split("\t")

        freq = 0.0
        r = ""
        target_score = 0.0
        for i in range(1,len(line)/2 + 1):
            if line[i*2-1] == target:
                target_score = float(line[i*2])
            if float(line[i*2]) > (freq):
                freq = float(line[i*2])
                r = line[i*2-1]

        #re = "0"   
        #if s >= t:
        #    re = "1"
        result.append(r)
        score.append(target_score)
    return result,score


def get_azp_feature_zp(zp_node,wl,trans):
    #azp feature 程序 created at 2015.12.24
    ifl = []
    node = zp_node

    #Pos
    after_index = node.index+1
    if after_index < len(wl):
        pos = wl[after_index].tag.split("-")[0]
        ifl.append("AfterPos%s:1"%pos)
    
        if pos.find("V") >= 0:
            if wl[after_index].word in trans:
                ifl.append("AfterTrans:1")

        after_after_index = after_index+1
        if after_after_index < len(wl):
            apos = wl[after_after_index].tag.split("-")[0]
            ifl.append("AfterAfterPos%s:1"%apos)
            
            ifl.append("12Pos%s:1"%(pos+apos))

    before_index = node.index-1
    if before_index >= 0:
        pos = wl[before_index].tag.split("-")[0]
        ifl.append("BeforePos%s:1"%pos)
   
        if pos.find("V") >= 0:
            if wl[before_index].word in trans:
                ifl.append("BeforeTrans:1")
 
        before_before_index = before_index-1
        if before_before_index >= 0:
            apos = wl[before_before_index].tag.split("-")[0]
            ifl.append("BeforeBeforePos%s:1"%apos)
            
            ifl.append("Neg12Pos%s:1"%(pos+apos))
            
    if after_index < len(wl):
        if before_index >= 0:
            ab_pos = wl[after_index].tag.split("-")[0] + wl[before_index].tag.split("-")[0]
            ifl.append("AFPos%s:1"%ab_pos)
    
    lowest_IP = None
    ip_node = node 
    while ip_node:
        if ip_node.tag.find("IP") >= 0:
            lowest_IP = ip_node
            break
        ip_node = ip_node.parent

    if lowest_IP:
        
        IP_child = lowest_IP.child    
        find_sub = False
        find_obj = False
        find_VP = False
        
        #如果找到VP，VP前边的NP是sub，后边的NP是obj
        if len(IP_child) > 0:
            for c in IP_child:
                if c.tag.find("NP") >= 0:
                    if not find_VP:
                        find_sub = True
                    else:
                        find_obj = True
                if c.tag.find("VP") >= 0:
                    find_VP = True
        
        leafs = lowest_IP.get_leaf()
        if leafs.index(node) == 0:
            ifl.append("FirstIP:1")
 
            if not find_sub:
                ifl.append("FirstInSubjectlessIP:1")

    wlw = node.left
    if wlw:
        if wlw.tag.find("PU") >= 0:
            VP_node = node
            while VP_node:
                if VP_node.tag.find("VP") >= 0:
                    ifl.append("FirstVPByPun:1")
                VP_node = VP_node.parent
        
    first_gap = "1"
    for i in range(node.index):
        if wl[i].word == "*pro*":
            first_gap = "0"
            break
    ifl.append("FirstGap:%s"%first_gap)
    
    wlw = node.left
    wr = node.right
    if wlw and wr:
        pub_node = wlw.get_pub_node(wr)
        if pub_node:
            wlf = wlw.parent
            wrf = wr.parent
            
            
            if wlf:
                if wlf.tag.find("NP") >= 0:
                    ifl.append("PlisNP:1")
            if wrf:
                if wrf.tag.find("VP") >= 0:
                    ifl.append("PrisVP:1")
            if wlf:
                if wrf:
                    if wlf.tag.find("NP") >= 0:
                        if wrf.tag.find("VP") >= 0:
                            ifl.append("PlNPandPrVP:1")

            ifl.append("Pub_node_%s:1"%pub_node.tag.split("-")[0])

    if node.parent.tag.find("VP") >= 0:
        ifl.append("PisVP:1")
   
    if node.parent.tag.find("NP") >= 0:
        ifl.append("PisNP:1")
     
    nf = node
    while nf:
        if nf.tag.find("NP") >= 0:
            ifl.append("HasAncNP:1")
            break
        nf = nf.parent

    nf = node
    while nf:
        if nf.tag.find("VP") >= 0:
            ifl.append("HasAncVP:1")
            break
        nf = nf.parent

    nf = node
    z_has_CP = "0"
    while nf:
        if nf.tag.find("CP") >= 0:
            ifl.append("HasAncCP:1")
            z_has_CP = "1"
            CP_node = nf
            break
        nf = nf.parent
    
    if wlw:
        if wlw.tag == "PU":
            ifl.append("LeftComma:1")

    tag = node.parent.tag.split("-") 
    z_gram_role = "0"
    z_Headline = "0"
    if len(tag) == 2:
        if tag[1] == "SBJ":
            z_gram_role = "1"
        if tag[1] == "HLN":
            z_Headline = "1"
    if z_gram_role == "1":
        ifl.append("ZGramRole:1")
    if z_Headline == "1":
        ifl.append("ZHeadLine:1")

    ifl.append("ZGramRole%s:1"%tag[0])

    #从句:有CP ancestor S
    #独立从句:有CP CP下有IP I
    #主句:直接有根节点的IP  M
    #其他:无CP，有非根节点的IP X
    z_clause = ""
    if z_has_CP == "1":
        z_clause = "S"
        father = node.parent
        while father:
            if father.tag.startswith("IP"):
                z_clause = "I"
                break
            if father == CP_node:
                break
            father = father.parent 
    else:
        z_clause = "M"
        father = node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent: #非根节点
                    z_clause = "X"
                    break
            father = father.parent
    #ifl.append("z_clause:%s"%z_clause)
    ifl.append("ZClause%s:1"%z_clause)

    return ifl

def get_azp_feature_MaxEnt(zps,candidates_list):
    fl = []
    for zp in zps:
        if not zp.is_zp:
            print >> sys.stderr,"No zp,",zp.head_index
            continue
        ifl = []
        node = zp.nodes[0]
        is_azp = "0"
        di = len(candidates_list)
        for candidates in candidates_list:
            di -= 1
            for candidate in candidates:
                if di == 0:
                    if candidate.head > node.index:
                        break 
                if zp.res(candidate):
                    is_azp = "1"
                    break
            if is_azp == "1":
                break     
        #ifl.append("is_azp:%s"%is_azp)
        ifl.append("%s"%is_azp)

        first_gap = "0"
        zp_index = zps.index(zp)
        if zp_index == 0:
            first_gap = "1"
        ifl.append("FirstGap:%s"%first_gap)
        
        wl = node.left
        wr = node.right
        if wl and wr:
            pub_node = wl.get_pub_node(wr)
            if pub_node:
                wlf = wl
                while wlf:
                    if pub_node.has_child(wlf):
                        break
                    wlf = wlf.parent
                wrf = wr
                while wrf:
                    if pub_node.has_child(wrf):
                        break
                    wrf = wrf.parent
                if wlf:
                    if wlf.tag.find("NP") >= 0:
                        ifl.append("PlisNP:1")
                if wrf:
                    if wrf.tag.find("VP") >= 0:
                        ifl.append("PrisVP:1")
                if wlf:
                    if wrf:
                        if wlf.tag.find("NP") >= 0:
                            if wrf.tag.find("VP") >= 0:
                                ifl.append("PlNPandPrVP:1")

        if node.parent.tag.find("VP") >= 0:
            ifl.append("PisVP:1")
        
        nf = node
        while nf:
            if nf.tag.find("NP") >= 0:
                ifl.append("HasAncNP:1")
                break
            nf = nf.parent

        nf = node
        while nf:
            if nf.tag.find("VP") >= 0:
                ifl.append("HasAncVP:1")
                break
            nf = nf.parent

        nf = node
        z_has_CP = "0"
        while nf:
            if nf.tag.find("CP") >= 0:
                ifl.append("HasAncCP:1")
                z_has_CP = "1"
                CP_node = nf
                break
            nf = nf.parent
        
        if first_gap == "1":
            ifl.append("LeftCommaNA:1")
        else:
            if wl:
                if wl.tag == "PU":
                    ifl.append("LeftComma:1")

        tag = node.parent.tag.split("-") 
        z_gram_role = "0"
        z_Headline = "0"
        if len(tag) == 2:
            if tag[1] == "SBJ":
                z_gram_role = "1"
            if tag[1] == "HLN":
                z_Headline = "1"
        if z_gram_role == "1":
            ifl.append("ZGramRole:1")
        if z_Headline == "1":
            ifl.append("ZHeadLine:1")

        #ifl.append("ZGramRole%s:1"%tag)

        #从句:有CP ancestor S
        #独立从句:有CP CP下有IP I
        #主句:直接有根节点的IP  M
        #其他:无CP，有非根节点的IP X
        z_clause = ""
        if z_has_CP == "1":
            z_clause = "S"
            father = node.parent
            while father:
                if father.tag.startswith("IP"):
                    z_clause = "I"
                    break
                if father == CP_node:
                    break
                father = father.parent 
        else:
            z_clause = "M"
            father = node.parent
            while father:
                if father.tag.startswith("IP"):
                    if father.parent: #非根节点
                        z_clause = "X"
                        break
                father = father.parent
        #ifl.append("z_clause:%s"%z_clause)
        ifl.append("ZClause%s:1"%z_clause)

        fl.append(ifl)
    return fl


def get_azp_feature_new(zps,candidates_list):
    fl = []
    for zp in zps:
        if not zp.is_zp:
            continue
        ifl = []
        node = zp.nodes[0]
        is_azp = "0"
        for candidates in candidates_list:
            for candidate in candidates:
                if zp.res(candidate):
                    is_azp = "1" 
                    break
            if is_azp == "1":
                break     
        ifl.append("is_azp:%s"%is_azp)

        first_gap = "0"
        zp_index = zps.index(zp)
        if zp_index == 0:
            first_gap = "1"
        if first_gap == "1":
            ifl.append("FirstGap")
        
        wl = node.left
        wr = node.right
        if wl and wr:
            pub_node = wl.get_pub_node(wr)
            if pub_node:
                wlf = wl
                while wlf:
                    if pub_node.has_child(wlf):
                        break
                    wlf = wlf.parent
                wrf = wr
                while wrf:
                    if pub_node.has_child(wrf):
                        break
                    wrf = wrf.parent
                if wlf:
                    if wlf.tag.find("NP") >= 0:
                        ifl.append("PlisNP")
                if wrf:
                    if wrf.tag.find("VP") >= 0:
                        ifl.append("PrisVP")
                if wlf:
                    if wrf:
                        if wlf.tag.find("NP") >= 0:
                            if wrf.tag.find("VP") >= 0:
                                ifl.append("PlNPandPrVP")

        if node.parent.tag.find("VP") >= 0:
            ifl.append("PisVP")
        
        nf = node
        while nf:
            if nf.tag.find("NP") >= 0:
                ifl.append("HasAncNP")
                break
            nf = nf.parent

        nf = node
        while nf:
            if nf.tag.find("VP") >= 0:
                ifl.append("HasAncVP")
                break
            nf = nf.parent

        nf = node
        z_has_CP = "0"
        while nf:
            if nf.tag.find("CP") >= 0:
                ifl.append("HasAncCP")
                z_has_CP = "1"
                CP_node = nf
                break
            nf = nf.parent
        
        if first_gap == "1":
            ifl.append("NA")
        else:
            if wl:
                if wl.tag == "PU":
                    ifl.append("LeftComma")

        tag = node.parent.tag.split("-") 
        z_gram_role = "0"
        z_Headline = "0"
        if len(tag) == 2:
            if tag[1] == "SBJ":
                z_gram_role = "1"
            if tag[1] == "HLN":
                z_Headline = "1"
        if z_gram_role == "1":
            ifl.append("ZGramRole")
        if z_Headline == "1":
            ifl.append("ZHeadLine")

        #从句:有CP ancestor S
        #独立从句:有CP CP下有IP I
        #主句:直接有根节点的IP  M
        #其他:无CP，有非根节点的IP X
        z_clause = ""
        if z_has_CP == "1":
            z_clause = "S"
            father = node.parent
            while father:
                if father.tag.startswith("IP"):
                    z_clause = "I"
                    break
                if father == CP_node:
                    break
                father = father.parent 
        else:
            z_clause = "M"
            father = node.parent
            while father:
                if father.tag.startswith("IP"):
                    if father.parent: #非根节点
                        z_clause = "X"
                        break
                father = father.parent
        #ifl.append("z_clause:%s"%z_clause)
        ifl.append("ZClause%s"%z_clause)

        fl.append(ifl)
    return fl

def get_res_feature_svm(zps,candidates_list,wl,lm):
    fl = []

    for zp in zps:
        zp_node = zp.nodes[0]

        if not (zp_node.word == "*pro*" or zp_node.word == "*PRO*" or zp_node.word == "*add*"):
            continue
        sentence_dis = len(candidates_list)
        for candidates in candidates_list:
            sentence_dis -= 1
            for candidate in candidates:

                o = []
                for n in candidate.nodes:
                    o.append(n.word)
                candidate_word = " ".join(o)
    

                ifl = []
                res = "0"
                if zp.res(candidate):
                    res = "1"
                ifl.append("%s"%res)

                dist_sentence = "1"
                if sentence_dis == 0:
                    dist_sentence = "0"
                ifl.append("1:%s"%dist_sentence)
                #ifl.append("dist_sentence:%s"%dist_sentence)
                
                dist_seg = "1"
                if sentence_dis == 0:
                    dist_seg = "0"
                    if zp_node.index <= candidate.head:
                        for node in wl[zp_node.index:candidate.head]:
                            if node.tag == "PU":
                                dist_seg = "1"
                                break
                    else:
                        if zp_node.index >= candidate.tail:
                            for node in wl[zp_node.index:candidate.head]:
                                if node.tag == "PU":
                                    dist_seg = "1"
                                    break
                #ifl.append("dist_seg:%s"%dist_seg)
                ifl.append("2:%s"%dist_seg)

                first_NP = "0"
                if sentence_dis == 0:
                    if candidate.tail <= zp_node.index:
                        first_NP = "1"
                    for i in range(candidate.tail+1,zp_node.index):
                        node = wl[i]
                        while True:
                            if node.tag.startswith("NP"):
                                first_NP = "0"
                                break
                            node = node.parent
                            if not node:
                                break
                        if first_NP == "0":
                            break
                #ifl.append("first_NP:%s"%first_NP)
                ifl.append("3:%s"%first_NP)
                #if first_NP == "1":
                #    ifl.append("FirstNP")

                #father_zp = zp_node.parent.tag
                #ifl.append("father_zp:%s"%father_zp)


                #For zp
                #找NP ancestor
                NP_node = None
                father = zp_node.parent
                while father:
                    if father.tag.startswith("NP"):
                        NP_node = father
                        break
                    father = father.parent
                z_has_anc_NP = "0"
                if NP_node:
                    z_has_anc_NP = "1"
                ifl.append("4:%s"%z_has_anc_NP)
                #ifl.append("z_has_anc_NP:%s"%z_has_anc_NP)

                z_has_NP_in_IP = "0"
                if NP_node:
                    father = zp_node.parent
                    while father:
                        if father.tag.startswith("IP"): 
                            if father.has_child(NP_node):
                                z_has_NP_in_IP = "1"
                            break
                        father = father.parent
                #ifl.append("z_has_NP_in_IP:%s"%z_has_NP_in_IP)
                ifl.append("5:%s"%z_has_NP_in_IP)

                VP_node = None
                z_has_VP = "0"
                father = zp_node.parent
                while father:
                    if father.tag.startswith("VP"):
                        VP_node = father
                        z_has_VP = "1"
                        break
                    father = father.parent
                #ifl.append("z_has_VP:%s"%z_has_VP)
                ifl.append("6:%s"%z_has_VP)

                z_has_VP_in_IP = "0"
                if VP_node:
                    father = zp_node.parent
                    while father:
                        if father.tag.startswith("IP"): 
                            if father.has_child(VP_node):
                                z_has_VP_in_IP = "1"
                            break
                        father = father.parent
                #ifl.append("z_has_VP_in_IP:%s"%z_has_VP_in_IP)
                ifl.append("7:%s"%z_has_VP_in_IP)
 
                CP_node = None
                z_has_CP = "0"
                father = zp_node.parent
                while father:
                    if father.tag.startswith("CP"):
                        CP_node = father
                        z_has_CP = "1"
                        break
                    father = father.parent
                #ifl.append("z_has_CP:%s"%z_has_CP)
                ifl.append("8:%s"%z_has_CP)
               
                tags = zp_node.parent.tag.split("-") 
                z_gram_role = "0"
                z_Headline = "0"
                if len(tags) == 2:
                    if tags[1] == "SBJ":
                        z_gram_role = "1"
                    if tags[1] == "HLN":
                        z_Headline = "1"
                ifl.append("9:%s"%z_gram_role)
                #ifl.append("z_gram_role:%s"%z_gram_role)
                ifl.append("10:%s"%z_Headline)
                #ifl.append("z_Headline:%s"%z_Headline)
               
                zp_index = zps.index(zp)

                z_first_ZP = "0"
                if zp_index == 0:
                    z_first_ZP = "1"
                #ifl.append("z_first_ZP:%s"%z_first_ZP)
                ifl.append("11:%s"%z_first_ZP)

                z_last_ZP = "0"
                if zp_index == len(zps) - 1:
                    z_last_ZP = "1"
                #ifl.append("z_last_ZP:%s"%z_last_ZP)
                ifl.append("12:%s"%z_last_ZP)

                #从句:有CP ancestor S
                #独立从句:有CP CP下有IP I
                #主句:直接有根节点的IP  M
                #其他:无CP，有非根节点的IP X
                z_clause = ""
                if z_has_CP == "1":
                    z_clause = "1"
                    father = zp_node.parent
                    while father:
                        if father.tag.startswith("IP"):
                            z_clause = "2"
                            break
                        if father == CP_node:
                            break
                        father = father.parent 
                else:
                    z_clause = "3"
                    father = zp_node.parent
                    while father:
                        if father.tag.startswith("IP"):
                            if father.parent: #非根节点
                                z_clause = "4"
                                break
                        father = father.parent
                #ifl.append("z_clause:%s"%z_clause)
                ifl.append("13:%s"%z_clause)

   
#Candidate 
                candi_node = candidate.nodes[0] #拿第一个node取特征
                father_candi = candi_node.parent.tag
                #ifl.append("father_candi:%s"%father_candi)

                #找NP ancestor
                NP_node = None
                father = candi_node.parent
                while father:
                    if father.tag.startswith("NP"):
                        NP_node = father
                        break
                    father = father.parent
                candi_has_NP_in_IP = "0"
                if NP_node:
                    father = candi_node.parent
                    while father:
                        if father.tag.startswith("IP"): 
                            if father.has_child(NP_node):
                                candi_has_NP_in_IP = "1"
                            break
                        father = father.parent
                ifl.append("14:%s"%candi_has_NP_in_IP)
                #ifl.append("candi_has_NP_in_IP:%s"%candi_has_NP_in_IP)

                VP_node = None
                candi_has_VP = "0"
                father = candi_node.parent
                while father:
                    if father.tag.startswith("VP"):
                        VP_node = father
                        candi_has_VP = "1"
                        break
                    father = father.parent
                #ifl.append("candi_has_VP:%s"%candi_has_VP)
                ifl.append("15:%s"%candi_has_VP)

                candi_has_VP_in_IP = "0"
                if VP_node:
                    father = candi_node.parent
                    while father:
                        if father.tag.startswith("IP"): 
                            if father.has_child(VP_node):
                                candi_has_VP_in_IP = "1"
                            break
                        father = father.parent
                #ifl.append("candi_has_VP_in_IP:%s"%candi_has_VP_in_IP)
                ifl.append("16:%s"%candi_has_VP_in_IP)
 
                CP_node = None
                candi_has_CP = "0"
                father = candi_node.parent
                while father:
                    if father.tag.startswith("CP"):
                        CP_node = father
                        candi_has_CP = "1"
                        break
                    father = father.parent
                #ifl.append("candi_has_CP:%s"%candi_has_CP)
                ifl.append("17:%s"%candi_has_CP)
               
                tags = candi_node.parent.tag.split("-") 
                candi_gram_role = "0"
                candi_ADV = "0"
                candi_TMP = "0"
                candi_PN = "0"
                candi_Headline = "0"
                if len(tags) == 2:
                    if tags[1] == "SBJ":
                        candi_gram_role = "1"
                    elif tags[1] == "OBJ":
                        candi_gram_role = "2"
                    if tags[1] == "ADV":
                        candi_ADV = "1"
                    if tags[1] == "TMP":
                        candi_TMP = "1"
                    if tags[1] == "PN":
                        candi_PN = "1"
                    if tags[1] == "HLN":
                        candi_Headline = "1"
                #ifl.append("candi_gram_role:%s"%candi_gram_role)
                ifl.append("18:%s"%candi_gram_role)
                #ifl.append("candi_ADV:%s"%candi_ADV)
                ifl.append("19:%s"%candi_ADV)
                #ifl.append("candi_TMP:%s"%candi_TMP)
                ifl.append("20:%s"%candi_TMP)
                #ifl.append("candi_PN:%s"%candi_PN)
                ifl.append("21:%s"%candi_PN)
                #ifl.append("candi_Headline:%s"%candi_Headline)
                ifl.append("22:%s"%candi_Headline)

                #从句:有CP ancestor S
                #独立从句:有CP CP下有IP I
                #主句:直接有根节点的IP  M
                #其他:无CP，有非根节点的IP X
                candi_clause = "0"
                if candi_has_CP == "1":
                    candi_clause = "1"
                    father = candi_node.parent
                    while father:
                        if father.tag.startswith("IP"):
                            candi_clause = "2"
                            break
                        if father == CP_node:
                            break
                        father = father.parent 
                else:
                    candi_clause = "3"
                    father = candi_node.parent
                    while father:
                        if father.tag.startswith("IP"):
                            if father.parent: #非根节点
                                candi_clause = "4"
                                break
                        father = father.parent
                #ifl.append("candi_clause:%s"%candi_clause)
                ifl.append("23:%s"%candi_clause)

#                ifl.append("zp:%d"%zp_node.index)
#                ifl.append("word:%s"%candidate_word)
                sibling_np_vp = "0"
                if not sentence_dis == 0:
                    sibling_np_vp = "0"
                else:
                    if abs(zp_node.index - candidate.tail) == 1:
                        sibling_np_vp = "1"
                    else:
                        if abs(zp_node.index - candidate.head) == 1:
                            sibling_np_vp = "1"
                        else:
                            if abs(zp_node.index-candidate.head) == 2:
                                if zp_node.index < candidate.head:
                                    if wl[zp_node.index+1].tag == "PU":
                                        sibling_np_vp = "1"
                            elif abs(zp_node.index-candidate.tail) == 2:
                                if candidate.tail < zp_node.index:
                                    if wl[zp_node.index-1].tag == "PU":
                                        sibling_np_vp = "1"
                #ifl.append("sibling_np_vp:%s"%sibling_np_vp)
                ifl.append("24:%s"%sibling_np_vp)

                left_word = ""
                if zp_node.left:
                    left_word = zp_node.left.word
                right_word = ""
                if zp_node.right:
                    left_word = zp_node.right.word
                score = lm.sentence_probability("%s %s %s"%(left_word,candidate_word,right_word))
                #print "%s %s %s"%(left_word,candidate_word,right_word)
                ifl.append("25:%.3f"%(score/50.0))

                gram_match = "0"
                if not candi_gram_role == "0":
                    if candi_gram_role == z_gram_role:
                        gram_match = "1"
                ifl.append("26:%s"%gram_match)
                 
                zp_node_tree = ""
                tree_node = zp_node
                while tree_node:
                    zp_node_tree = "( "+tree_node.tag+" "+zp_node_tree+" )"
                    tree_node = tree_node.parent

                candi_node_tree = ""
                tree_node = candi_node
                while tree_node:
                    candi_node_tree = "( "+tree_node.tag+" "+candi_node_tree+" )"
                    tree_node = tree_node.parent
                s,sent_tree1,sent_tree2=getsim_by_str(zp_node_tree,candi_node_tree)
                ifl.append("27:%.3f"%(s*5))
                #candi_node

                fl.append(ifl)
    return fl

def get_head_verb(index,wl):
    father = wl[index].parent
    while father:
        leafs = father.get_leaf()
        for ln in leafs:
            if ln.tag.startswith("V"):
                return ln
        father = father.parent

    return None

def get_template(zp,candidate,wl_zp,wl_candi,HcPz,PcPz,HcP):

    ifl = []

    (zp_sentence_index,zp_index) = zp
    (candi_sentence_index,candi_index_begin,candi_index_end) = candidate

    zp_node = wl_zp[zp_index]
    
    candi_node = wl_candi[candi_index_begin]
   
    tags = candi_node.parent.tag.split("-") 
    candi_gram_role = "0"
    candi_ADV = "0"
    candi_TMP = "0"
    candi_PN = "0"
    candi_Headline = "0"
    if len(tags) == 2:
        if tags[1] == "SBJ":
            candi_gram_role = "SBJJ"
        elif tags[1] == "OBJ":
            candi_gram_role = "OBJ"
        if tags[1] == "ADV":
            candi_ADV = "1"
        if tags[1] == "TMP":
            candi_TMP = "1"
        if tags[1] == "PN":
            candi_PN = "1"
        if tags[1] == "HLN":
            candi_Headline = "1"

    candi_head_verb = get_head_verb(candi_index_begin,wl_candi)
    zp_head_verb = get_head_verb(zp_index,wl_zp)
    candi_head = wl_candi[candi_index_end]

    hc = "None"
    pc = "None"
    pz = "None"

    if candi_head:
        hc = candi_head.word
    if zp_head_verb:
        pz = zp_head_verb.word
    if candi_head_verb:
        pc = candi_head_verb.word

    tags = candi_node.parent.tag.split("-")
    candi_gram_role = "None"
    if len(tags) == 2:
        if tags[1] == "SBJ":
            candi_gram_role = "SBJ"
        elif tags[1] == "OBJ":
            candi_gram_role = "OBJ"
    gc = candi_gram_role

    punc = "None"
    for i in range(len(wl_zp)-1,zp_index,-1):
        if wl_zp[i].tag.find("PU") >= 0:
            punc = wl_zp[i].word
            break 

    hcpz = "%s_%s"%(hc,pz)
    has_hcpz = "0"
    if hcpz in HcPz:
        has_hcpz = "1"
    ifl.append("Head_c+Pred_z:%s"%hcpz)

    pc_pz_same = "0"
    if pc == pz:
        if candi_gram_role == "SBJ":
            pc_pz_same = "1"
        elif candi_gram_role == "OBJ":
            pc_pz_same = "1"
        else:
            pz_pz_same = "2"


    pcpz = "%s_%s_%s"%(pc,pz,gc)
    has_pcpz = "0"
    if pcpz in PcPz:
        has_pcpz = "1"
    ifl.append("Pred_c+Pred_z+Gram_c:%s"%pcpz)

    hcp = "%s_%s"%(hc,punc)
    has_hcp = "0"
    if hcp in HcP:
        has_hcp = "1"
    ifl.append("Head_c+Punc:%s"%hcp)

    return ifl


def get_res_feature_NN(zp,candidate,wl_zp,wl_candi,HcPz,PcPz,HcP):

    ifl = []

    (zp_sentence_index,zp_index) = zp
    (candi_sentence_index,candi_index_begin,candi_index_end) = candidate

    zp_node = wl_zp[zp_index]

    sentence_dis = zp_sentence_index - candi_sentence_index

    tmp_ones = [0]*3
    tmp_ones[sentence_dis] = 1 
    ifl += tmp_ones
   
    #dist_seg = 10
    #if sentence_dis == 0:
    #    if zp_index >= candi_index_end:
    #        dist_seg = 0
    #        for node in wl_zp[candi_index_end:zp_index]:
    #            if node.tag == "PU":
    #                dist_seg += 1
        
    #ifl.append("dist_seg_%d:1"%dist_seg)

    closest_NP = 0
    if sentence_dis == 0:
        if candi_index_end <= zp_index:
            closest_NP = 1
        for i in range(candi_index_end+1,zp_index):
            node = wl_zp[i]
            while True:
                if node.tag.startswith("NP"):
                    closest_NP = 0
                    break
                node = node.parent
                if not node:
                    break
            if closest_NP == 0:
                break

    tmp_ones = [0]*2
    tmp_ones[closest_NP] = 1
    ifl += tmp_ones

    #For zp
    #找NP ancestor
    NP_node = None
    father = zp_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    z_has_anc_NP = 0
    if NP_node:
        z_has_anc_NP = 1
    #ifl.append("4:%s"%z_has_anc_NP)
    tmp_ones = [0]*2
    tmp_ones[z_has_anc_NP] = 1
    ifl += tmp_ones

    z_has_NP_in_IP = 0
    if NP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(NP_node):
                    z_has_NP_in_IP = 1
                break
            father = father.parent

    tmp_ones = [0]*2
    tmp_ones[z_has_NP_in_IP] = 1
    ifl += tmp_ones

    VP_node = None
    z_has_VP = 0
    father = zp_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            z_has_VP = 1
            break
        father = father.parent
    
    tmp_ones = [0]*2
    tmp_ones[z_has_VP] = 1
    ifl += tmp_ones

    z_has_VP_in_IP = 0
    if VP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(VP_node):
                    z_has_VP_in_IP = 1
                break
            father = father.parent
    
    tmp_ones = [0]*2
    tmp_ones[z_has_VP_in_IP] = 1
    ifl += tmp_ones

 
    CP_node = None
    z_has_CP = 0
    father = zp_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            z_has_CP = 1
            break
        father = father.parent

    tmp_ones = [0]*2
    tmp_ones[z_has_CP] = 1
    ifl += tmp_ones
   
    tags = zp_node.parent.tag.split("-") 
    z_gram_role = 0
    z_Headline = 0
    if len(tags) == 2:
        if tags[1] == "SBJ":
            z_gram_role = 1
        if tags[1] == "HLN":
            z_Headline = 1

    tmp_ones = [0]*2
    tmp_ones[z_gram_role] = 1
    ifl += tmp_ones

    tmp_ones = [0]*2
    tmp_ones[z_Headline] = 1
    ifl += tmp_ones
  
    first_zp = 1
    for i in range(zp_index):
        if wl_zp[i].word == "*pro*":
            first_zp = 0
            break
    tmp_ones = [0]*2
    tmp_ones[first_zp] = 1
    ifl += tmp_ones

    last_zp = 1
    for i in range(zp_index+1,len(wl_zp)):
        if wl_zp[i].word == "*pro*":
            last_zp = 0
            break
    tmp_ones = [0]*2
    tmp_ones[last_zp] = 1
    ifl += tmp_ones

    #从句:有CP ancestor S
    #独立从句:有CP CP下有IP I
    #主句:直接有根节点的IP  M
    #其他:无CP，有非根节点的IP X
    z_clause = 0
    if z_has_CP == 1:
        z_clause = 1
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                z_clause = 2
                break
            if father == CP_node:
                break
            father = father.parent 
    else:
        z_clause = 3
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent: #非根节点
                    z_clause = 4
                    break
            father = father.parent

    tmp_ones = [0]*5
    tmp_ones[z_clause] = 1
    ifl += tmp_ones
   
#Candidate 
    #candi_node = candidate.nodes[0] #拿第一个node取特征
    candi_node = wl_candi[candi_index_begin] #拿第一个node取特征
    father_candi = candi_node.parent.tag
    #ifl.append("father_candi_%s:1"%father_candi)

    #找NP ancestor
    NP_node = None
    father = candi_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    candi_has_NP_in_IP = 0
    if NP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(NP_node):
                    candi_has_NP_in_IP = 1
                break
            father = father.parent
    tmp_ones = [0]*2
    tmp_ones[candi_has_NP_in_IP] = 1
    ifl += tmp_ones

    VP_node = None
    candi_has_VP = 0
    father = candi_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            candi_has_VP = 1
            break
        father = father.parent
    tmp_ones = [0]*2
    tmp_ones[candi_has_VP] = 1
    ifl += tmp_ones    

    candi_has_VP_in_IP = 0
    if VP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(VP_node):
                    candi_has_VP_in_IP = 1
                break
            father = father.parent
    tmp_ones = [0]*2
    tmp_ones[candi_has_VP_in_IP] = 1
    ifl += tmp_ones
 
    CP_node = None
    candi_has_CP = 0
    father = candi_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            candi_has_CP = 1
            break
        father = father.parent
    tmp_ones = [0]*2
    tmp_ones[candi_has_CP] = 1
    ifl += tmp_ones

    tags = candi_node.parent.tag.split("-") 
    candi_gram_role = 0
    candi_ADV = 0
    candi_TMP = 0
    candi_PN = 0
    candi_Headline = 0
    if len(tags) == 2:
        if tags[1] == "SBJ":
            candi_gram_role = 1
        elif tags[1] == "OBJ":
            candi_gram_role = 2
        if tags[1] == "ADV":
            candi_ADV = 1
        if tags[1] == "TMP":
            candi_TMP = 1
        if tags[1] == "PN":
            candi_PN = 1
        if tags[1] == "HLN":
            candi_Headline = 1
    tmp_ones = [0]*3
    tmp_ones[candi_gram_role] = 1
    ifl += tmp_ones

    tmp_ones = [0]*2
    tmp_ones[candi_ADV] = 1
    ifl += tmp_ones

    tmp_ones = [0]*2
    tmp_ones[candi_TMP] = 1
    ifl += tmp_ones

    tmp_ones = [0]*2
    tmp_ones[candi_PN] = 1
    ifl += tmp_ones

    tmp_ones = [0]*2
    tmp_ones[candi_Headline] = 1
    ifl += tmp_ones

    #从句:有CP ancestor S
    #独立从句:有CP CP下有IP I
    #主句:直接有根节点的IP  M
    #其他:无CP，有非根节点的IP X
    candi_clause = 0
    if candi_has_CP == 1:
        candi_clause = 1
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                candi_clause = 2
                break
            if father == CP_node:
                break
            father = father.parent 
    else:
        candi_clause = 3
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent: #非根节点
                    candi_clause = 4
                    break
            father = father.parent
    tmp_ones = [0]*5
    tmp_ones[candi_clause] = 1
    ifl += tmp_ones


    sibling_np_vp = 0
    if not sentence_dis == 0:
        sibling_np_vp = 0
    else:
        if abs(zp_index - candi_index_end) == 1:
            sibling_np_vp = 1
        else:
            if abs(zp_index - candi_index_begin) == 1:
                sibling_np_vp = 1
            else:
                if abs(zp_index - candi_index_begin) == 2:
                    if zp_index < candi_index_begin:
                        if wl_zp[zp_index+1].tag == "PU":
                            sibling_np_vp = 1
                elif abs(zp_index-candi_index_end) == 2:
                    if candi_index_end < zp_index:
                        if wl_zp[zp_index-1].tag == "PU":
                            sibling_np_vp = 1
    tmp_ones = [0]*2
    tmp_ones[sibling_np_vp] = 1
    ifl += tmp_ones


    gram_match = 0
    if not candi_gram_role == 0:
        if candi_gram_role == z_gram_role:
            gram_match = 1

    tmp_ones = [0]*2
    tmp_ones[gram_match] = 1
    ifl += tmp_ones


    ##### return #####
#    return ifl

    #candi_node

    candi_head_verb = get_head_verb(candi_index_begin,wl_candi)
    zp_head_verb = get_head_verb(zp_index,wl_zp)
    candi_head = wl_candi[candi_index_end]

    hc = "None"
    pc = "None"
    pz = "None"

    if candi_head:
        hc = candi_head.word
    if zp_head_verb:
        pz = zp_head_verb.word
    if candi_head_verb:
        pc = candi_head_verb.word

    tags = candi_node.parent.tag.split("-")
    candi_gram_role = "None"
    if len(tags) == 2:
        if tags[1] == "SBJ":
            candi_gram_role = "SBJ"
        elif tags[1] == "OBJ":
            candi_gram_role = "OBJ"
    gc = candi_gram_role

    punc = "None"
    for i in range(len(wl_zp)-1,zp_index,-1):
        if wl_zp[i].tag.find("PU") >= 0:
            punc = wl_zp[i].word
            break 

    hcpz = "%s_%s"%(hc,pz)
    has_hcpz = "0"
    if hcpz in HcPz:
        has_hcpz = "1"
    #ifl.append("28:%s"%has_hcpz)

    pc_pz_same = 0
    if pc == pz:
        if candi_gram_role == "SBJ":
            pc_pz_same = 1
        elif candi_gram_role == "OBJ":
            pc_pz_same = 1
        else:
            pc_pz_same = 2

    tmp_ones = [0]*3
    tmp_ones[pc_pz_same] = 1
    ifl += tmp_ones

    #ifl.append("zp_pz_same_%s:1"%pc_pz_same)

    pcpz = "%s_%s_%s"%(pc,pz,gc)
    has_pcpz = "0"
    if pcpz in PcPz:
        has_pcpz = "1"

    hcp = "%s_%s"%(hc,punc)
    has_hcp = 0
    if hcp in HcP:
        has_hcp = 1
    
    tmp_ones = [0]*2
    tmp_ones[has_hcp] = 1
    ifl += tmp_ones
 
    ##### return #####
    return ifl




def get_res_feature_Max(zp,candidate,wl_zp,wl_candi,HcPz,PcPz,HcP):

    ifl = []

    (zp_sentence_index,zp_index) = zp
    (candi_sentence_index,candi_index_begin,candi_index_end) = candidate

    zp_node = wl_zp[zp_index]

    sentence_dis = zp_sentence_index - candi_sentence_index

    ifl.append("sentence_dis_%d:1"%sentence_dis)
    
    dist_seg = 10
    if sentence_dis == 0:
        if zp_index >= candi_index_end:
            dist_seg = 0
            for node in wl_zp[candi_index_end:zp_index]:
                if node.tag == "PU":
                    dist_seg += 1
        
    ifl.append("dist_seg_%d:1"%dist_seg)

    closest_NP = "0"
    if sentence_dis == 0:
        if candi_index_end <= zp_index:
            closest_NP = "1"
        for i in range(candi_index_end+1,zp_index):
            node = wl_zp[i]
            while True:
                if node.tag.startswith("NP"):
                    closest_NP = "0"
                    break
                node = node.parent
                if not node:
                    break
            if closest_NP == "0":
                break
    ifl.append("closest_NP:%s"%closest_NP)

    #For zp
    #找NP ancestor
    NP_node = None
    father = zp_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    z_has_anc_NP = "0"
    if NP_node:
        z_has_anc_NP = "1"
    #ifl.append("4:%s"%z_has_anc_NP)
    ifl.append("z_has_anc_NP:%s"%z_has_anc_NP)

    z_has_NP_in_IP = "0"
    if NP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(NP_node):
                    z_has_NP_in_IP = "1"
                break
            father = father.parent
    ifl.append("z_has_NP_in_IP:%s"%z_has_NP_in_IP)
    #ifl.append("5:%s"%z_has_NP_in_IP)

    VP_node = None
    z_has_VP = "0"
    father = zp_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            z_has_VP = "1"
            break
        father = father.parent
    ifl.append("z_has_VP:%s"%z_has_VP)
    #ifl.append("6:%s"%z_has_VP)

    z_has_VP_in_IP = "0"
    if VP_node:
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(VP_node):
                    z_has_VP_in_IP = "1"
                break
            father = father.parent
    ifl.append("z_has_VP_in_IP:%s"%z_has_VP_in_IP)
    #ifl.append("7:%s"%z_has_VP_in_IP)
 
    CP_node = None
    z_has_CP = "0"
    father = zp_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            z_has_CP = "1"
            break
        father = father.parent
    ifl.append("z_has_CP:%s"%z_has_CP)
    #ifl.append("8:%s"%z_has_CP)
   
    tags = zp_node.parent.tag.split("-") 
    z_gram_role = "0"
    z_Headline = "0"
    if len(tags) == 2:
        if tags[1] == "SBJ":
            z_gram_role = "1"
        if tags[1] == "HLN":
            z_Headline = "1"
    #ifl.append("9:%s"%z_gram_role)
    ifl.append("z_gram_role:%s"%z_gram_role)
    #ifl.append("10:%s"%z_Headline)
    ifl.append("z_Headline:%s"%z_Headline)
  
    first_zp = "1"
    for i in range(zp_index):
        if wl_zp[i].word == "*pro*":
            first_zp = "0"
            break
    ifl.append("z_first_ZP:%s"%first_zp)


    last_zp = "1"
    for i in range(zp_index+1,len(wl_zp)):
        if wl_zp[i].word == "*pro*":
            last_zp = "0"
            break
    ifl.append("z_last_ZP:%s"%last_zp)


    #从句:有CP ancestor S
    #独立从句:有CP CP下有IP I
    #主句:直接有根节点的IP  M
    #其他:无CP，有非根节点的IP X
    z_clause = ""
    if z_has_CP == "1":
        z_clause = "S"
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                z_clause = "I"
                break
            if father == CP_node:
                break
            father = father.parent 
    else:
        z_clause = "M"
        father = zp_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent: #非根节点
                    z_clause = "X"
                    break
            father = father.parent
    ifl.append("z_clause_%s:1"%z_clause)
    #ifl.append("13:%s"%z_clause)

   
#Candidate 
    #candi_node = candidate.nodes[0] #拿第一个node取特征
    candi_node = wl_candi[candi_index_begin] #拿第一个node取特征
    father_candi = candi_node.parent.tag
    ifl.append("father_candi_%s:1"%father_candi)

    #找NP ancestor
    NP_node = None
    father = candi_node.parent
    while father:
        if father.tag.startswith("NP"):
            NP_node = father
            break
        father = father.parent
    candi_has_NP_in_IP = "0"
    if NP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(NP_node):
                    candi_has_NP_in_IP = "1"
                break
            father = father.parent
    #ifl.append("14:%s"%candi_has_NP_in_IP)
    ifl.append("candi_has_NP_in_IP:%s"%candi_has_NP_in_IP)

    VP_node = None
    candi_has_VP = "0"
    father = candi_node.parent
    while father:
        if father.tag.startswith("VP"):
            VP_node = father
            candi_has_VP = "1"
            break
        father = father.parent
    ifl.append("candi_has_VP:%s"%candi_has_VP)
    #ifl.append("15:%s"%candi_has_VP)

    candi_has_VP_in_IP = "0"
    if VP_node:
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"): 
                if father.has_child(VP_node):
                    candi_has_VP_in_IP = "1"
                break
            father = father.parent
    ifl.append("candi_has_VP_in_IP:%s"%candi_has_VP_in_IP)
    #ifl.append("16:%s"%candi_has_VP_in_IP)
 
    CP_node = None
    candi_has_CP = "0"
    father = candi_node.parent
    while father:
        if father.tag.startswith("CP"):
            CP_node = father
            candi_has_CP = "1"
            break
        father = father.parent
    ifl.append("candi_has_CP:%s"%candi_has_CP)
    #ifl.append("17:%s"%candi_has_CP)
   
    tags = candi_node.parent.tag.split("-") 
    candi_gram_role = "0"
    candi_ADV = "0"
    candi_TMP = "0"
    candi_PN = "0"
    candi_Headline = "0"
    if len(tags) == 2:
        if tags[1] == "SBJ":
            candi_gram_role = "SBJJ"
        elif tags[1] == "OBJ":
            candi_gram_role = "OBJ"
        if tags[1] == "ADV":
            candi_ADV = "1"
        if tags[1] == "TMP":
            candi_TMP = "1"
        if tags[1] == "PN":
            candi_PN = "1"
        if tags[1] == "HLN":
            candi_Headline = "1"
    ifl.append("candi_gram_role_%s:1"%candi_gram_role)
    #ifl.append("18:%s"%candi_gram_role)
    ifl.append("candi_ADV:%s"%candi_ADV)
    #ifl.append("19:%s"%candi_ADV)
    ifl.append("candi_TMP:%s"%candi_TMP)
    #ifl.append("20:%s"%candi_TMP)
    ifl.append("candi_PN:%s"%candi_PN)
    #ifl.append("21:%s"%candi_PN)
    ifl.append("candi_Headline:%s"%candi_Headline)
    #ifl.append("22:%s"%candi_Headline)

    #从句:有CP ancestor S
    #独立从句:有CP CP下有IP I
    #主句:直接有根节点的IP  M
    #其他:无CP，有非根节点的IP X
    candi_clause = "0"
    if candi_has_CP == "1":
        candi_clause = "S"
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                candi_clause = "I"
                break
            if father == CP_node:
                break
            father = father.parent 
    else:
        candi_clause = "M"
        father = candi_node.parent
        while father:
            if father.tag.startswith("IP"):
                if father.parent: #非根节点
                    candi_clause = "X"
                    break
            father = father.parent
    ifl.append("candi_clause_%s:1"%candi_clause)


    sibling_np_vp = "0"
    if not sentence_dis == 0:
        sibling_np_vp = "0"
    else:
        if abs(zp_index - candi_index_end) == 1:
            sibling_np_vp = "1"
        else:
            if abs(zp_index - candi_index_begin) == 1:
                sibling_np_vp = "1"
            else:
                if abs(zp_index - candi_index_begin) == 2:
                    if zp_index < candi_index_begin:
                        if wl_zp[zp_index+1].tag == "PU":
                            sibling_np_vp = "1"
                elif abs(zp_index-candi_index_end) == 2:
                    if candi_index_end < zp_index:
                        if wl_zp[zp_index-1].tag == "PU":
                            sibling_np_vp = "1"
    ifl.append("sibling_np_vp:%s"%sibling_np_vp)
    #ifl.append("24:%s"%sibling_np_vp)

    gram_match = "0"
    if not candi_gram_role == "0":
        if candi_gram_role == z_gram_role:
            gram_match = "1"
    #ifl.append("26:%s"%gram_match)
    ifl.append("gram_match:%s"%gram_match)

    #candi_node

    candi_head_verb = get_head_verb(candi_index_begin,wl_candi)
    zp_head_verb = get_head_verb(zp_index,wl_zp)
    candi_head = wl_candi[candi_index_end]

    hc = "None"
    pc = "None"
    pz = "None"

    if candi_head:
        hc = candi_head.word
    if zp_head_verb:
        pz = zp_head_verb.word
    if candi_head_verb:
        pc = candi_head_verb.word

    tags = candi_node.parent.tag.split("-")
    candi_gram_role = "None"
    if len(tags) == 2:
        if tags[1] == "SBJ":
            candi_gram_role = "SBJ"
        elif tags[1] == "OBJ":
            candi_gram_role = "OBJ"
    gc = candi_gram_role

    punc = "None"
    for i in range(len(wl_zp)-1,zp_index,-1):
        if wl_zp[i].tag.find("PU") >= 0:
            punc = wl_zp[i].word
            break 

    hcpz = "%s_%s"%(hc,pz)
    has_hcpz = "0"
    if hcpz in HcPz:
        has_hcpz = "1"
    ifl.append("28:%s"%has_hcpz)

    pc_pz_same = "0"
    if pc == pz:
        if candi_gram_role == "SBJ":
            pc_pz_same = "1"
        elif candi_gram_role == "OBJ":
            pc_pz_same = "1"
        else:
            pz_pz_same = "2"

    ifl.append("zp_pz_same_%s:1"%pc_pz_same)

    pcpz = "%s_%s_%s"%(pc,pz,gc)
    has_pcpz = "0"
    if pcpz in PcPz:
        has_pcpz = "1"

    hcp = "%s_%s"%(hc,punc)
    has_hcp = "0"
    if hcp in HcP:
        has_hcp = "1"
    ifl.append("hcp:%s"%has_hcp)

    return ifl


def write_feature_file(filename,feature_list,sentence_index):
    index = 1
    f = open(filename,"w")
    for feature in feature_list:
        #out = "%d\t%s\t%s\n"%(index,feature[0]," ".join(feature[1:]))  
        out = "%d%d\t%s\t%s\n"%(sentence_index,index,feature[0]," ".join(feature[1:]))  
        #out = "%d%d\t%s\t%s\n"%(sentence_index,index,feature[0]," ".join(feature[1:]).replace("-","").replace("_",""))  
        f.write(out)
        index += 1 
    f.close()
def write_feature_file_MaxEnt(filename,feature_list,sentence_index):
    index = 1
    f = open(filename,"w")
    for feature in feature_list:
        #out = "%d\t%s\t%s\n"%(index,feature[0]," ".join(feature[1:]))  
        #out = "%d%d\t%s\t%s\n"%(sentence_index,index,feature[0]," ".join(feature[1:]))  
        out = "%s\t%s\n"%(feature[0]," ".join(feature[1:]))  
        #out = "%d%d\t%s\t%s\n"%(sentence_index,index,feature[0]," ".join(feature[1:]).replace("-","").replace("_",""))  
        #print out
        f.write(out)
        index += 1 
    f.close()

def write_feature_file_svm(filename,feature_list,sentence_index):
    index = 1
    f = open(filename,"w")
    for feature in feature_list:
        #out = "%d\t%s\t%s\n"%(index,feature[0]," ".join(feature[1:]))  
        out = "%s %s\n"%(feature[0]," ".join(feature[1:]))  
        f.write(out)
        index += 1 
    f.close()


def print_feature_svm(sentence_index,feature_list):
    index = 1
    for feature in feature_list:
        print "%s %s"%(feature[0]," ".join(feature[1:]))  

def print_feature(sentence_index,feature_list):
    index = 1
    for feature in feature_list:
        print "%s\t%s"%(feature[0]," ".join(feature[1:]))  
        #print "%d.%d\t%s\t%s"%(sentence_index,index,feature[0]," ".join(feature[1:]))  
        index += 1

def main():

    parse = open(sys.argv[2])
    sentence = open(sys.argv[1])

    lm = SRILm(open("./dict/model.srilm", "r"))

    trans = []
    f = open("./dict/trans.word")
    while True:
        line = f.readline()
        if not line:break
        trans.append(line.strip())
    f.close()

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

    Max = 3
    candidates_list = []
    dependency_list = []
    wl_list = []

    sentence_index = 0

    while True:
        line = parse.readline() 
        if not line:break
        line = line.strip()

        if line == "<<end>>":
            candidates_list = []
            wl_list = []
            dependency_list = []
            continue
        
        #if len(line) >= 1600:
        #    line = sentence.readline()
        #    continue
        nl,wl = buildTree.buildTree(line)

        dependency = {}

        '''
        #dependency
        fw = open("./dependency/p","w")
        fw.write(line.decode("utf8").encode("gbk").replace("*","").replace("-","").replace("TOP","")[1:-1].strip()+"\n")
        fw.close()

        
        cmd = "./go_depenend.sh > t"
        os.system(cmd)
        fr = open("./dependency/result")
        if len(fr.readlines()) < 1:
            line = sentence.readline()
            continue
        fr.close()

        fr = open("./dependency/result")
        index = 0
        while True:
            line = fr.readline()
            if not line:break
            line = line.strip()
            if not line:break
            line = line.strip().decode("gbk").encode("utf8").split("\t")
            #put in dependency 
            dependency[index] = int(line[2]) - 1
            index += 1
        '''


        #print_node_list(nl)
        #print_node_list(wl)
        line = sentence.readline()
        print >> sys.stderr,sentence_index,line.strip()
        sentence_index += 1
        #line = line.strip().split("\t")
        resolution_list_sentence = buildTree.get_resolution_info(line.strip())
        resolution_list_parse = buildTree.get_np(nl) 
        #print_node_list(wl)
        zps,candidates = buildTree.get_resolution_pair(resolution_list_sentence,resolution_list_parse,wl) 
        candidates_list.append(candidates)
        if len(candidates_list) > Max:
            candidates_list = candidates_list[1:]

        dependency_list.append(dependency)
        if len(dependency_list) > Max:
            dependency_list = dependency_list[1:]

        wl_list.append(wl)
        if len(wl_list) > Max:
            wl_list = wl_list[1:]

        '''
        print "######"
        print "ZP:"
        for zp in zpl:
            zp.show()
        print 
        print "Candidates:"
        for candi in candidatel:
            candi.show()
        print "######"
        '''
        #azp特征提取
        feature_list = get_azp_feature_MaxEnt_new(zps,candidates_list,wl,trans)
        print_feature(sentence_index,feature_list)
        '''
        for zp in zps:
            for candidate in candidates:
                res = "0"
                if zp.res(candidate):
                    res = "1"
                print "%s\tzp:%s candi:%s"%(res,toString(zp.nodes),toString(candidate.nodes))
        '''
        #feature_list = get_cluster_feature(zps,candidates_list,wl)
        #print_feature(sentence_index,feature_list)

        #feature_list = get_res_feature(zps,candidates_list,wl)
        #print_feature(sentence_index,feature_list)
        #feature_list = get_res_feature_svm(zps,candidates_list,wl,lm)
        #feature_list = get_res_feature_svm_new_auto(zps,candidates_list,wl,lm,dependency_list,wl_list,HcPz,PcPz,HcP,dependency)
        #feature_list = get_res_feature_svm_new(zps,candidates_list,wl,lm,dependency_list,wl_list,HcPz,PcPz,HcP,dependency)
        #feature_list = get_res_feature_Max(zps,candidates_list,wl,wl_list,lm)
        #feature_list = get_res_feature_Max(zps,candidates_list,wl,wl_list,lm,dependency_list,dependency,HcPz,PcPz,HcP)
        #print_feature(sentence_index,feature_list)

#def get_res_feature_Max(zps,candidates_list,wl,wl_list):
        #print_feature_svm(sentence_index,feature_list)


if __name__ == "__main__":
    main()
