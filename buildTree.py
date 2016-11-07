#coding=utf8
import get_dir
import os
import sys
import re
import parse_analysis
from subprocess import *

def dif(l1,l2):
    if not (len(l1) == len(l2)):
        return True
    for i in range(len(l1)):
        if not (l1[i] == l2[i]):
            return True
    return False

def is_pro(leaf_nodes):
    if len(leaf_nodes) == 1:
        if leaf_nodes[0].word == "*pro*":
            return True
    return False


def get_info_from_file_system(file_name,parser_in,parser_out,MAX=2):

    pattern = re.compile("(\d+?)\ +(.+?)$")
    pattern_zp = re.compile("(\d+?)\.(\d+?)\-(\d+?)\ +(.+?)$")

    total = 0

    inline = "new"
    f = open(file_name)
    
    sentence_num = 0

    '''
    ################################################################################
    # nodes_info: (dict) 存放着对应sentence_index下的每个sentence的 nl 和 wl #
    #    ------------- nodes_info[sentence_index] = (nl,wl)                   #
    # candi: (dict) 存放着sentence_index下的每个candidate                          #
    #    ------------- candi[sentence_index] = list of (begin_index,end_index)      #
    # zps:  (list)  存放着对应file下的每个zp                                       #
    #    ------------- item : (sentence_index,zp_index)
    # azps:  (list)  存放着对应file下的每个azp                                       #
    #    ------------- 每个item 对应着 (sentence_index,zp_index,antecedents=[],is_azp)
    #   -------------  antecedents - (sentence_index,begin_word_index,end_word_index)
    ################################################################################
    '''
    nodes_info = {}   
    candi = {}
    zps = []
    azps = []

    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()

        if line == "Leaves:":
            while True:
                inline = f.readline()
                if inline.strip() == "":break
                inline = inline.strip()
                match = pattern.match(inline)
                if match:
                    word = match.groups()[1]
                    #if word == "*pro*":
                    #    print word
                    #if word.find("*") < 0:
                    #    print word
            sentence_num += 1
    
        elif line == "Tree:":
            candi[sentence_num] = []
            nodes_info[sentence_num] = None
            parse_info = ""
            inline = f.readline()
            while True:
                inline = f.readline()
                if inline.strip("\n") == "":break
                parse_info = parse_info + " " + inline.strip()    
            parse_info = parse_info.strip()            
            nl,wl = parse_analysis.buildTree(parse_info)

            pw = []
            for word in wl:
                pw.append(word.word)

            parser_in.write(" ".join(pw)+"\n")
            parse_info = parser_out.readline().strip()
            parse_info = "(TOP"+parse_info[1:-1]+")"
            nl,wl = parse_analysis.buildTree(parse_info)

            nodes_info[sentence_num] = (nl,wl)

            for node in nl:
                if (node.tag.find("NP") >= 0) and (node.tag.find("DNP") < 0):
                    if (node.tag.find("NP") >= 0) and (node.tag.find("DNP") < 0):
                        if not (node == node.parent.child[0]):
                            continue
                    leaf_nodes = node.get_leaf()
                    if is_pro(leaf_nodes):
                        continue

                    candi[sentence_num].append((leaf_nodes[0].index,leaf_nodes[-1].index))
                    total += 1
            for node in wl:
                if node.word == "*pro*":
                    zps.append((sentence_num,node.index))  
 
        elif line.startswith("Coreference chain"):
            first = True
            res_info = None
            last_index = 0
            antecedents = []

            while True:
                inline = f.readline()
                if not inline:break
                if inline.startswith("----------------------------------------------------------------------------------"):
                    break
                inline = inline.strip()
                if len(inline) <= 0:continue
                if inline.startswith("Chain"):
                    first = True
                    res_info = None
                    last_index = 0
                    antecedents = []
                else:
                    match = pattern_zp.match(inline)
                    if match:
                        sentence_index = int(match.groups()[0])
                        begin_word_index = int(match.groups()[1])
                        end_word_index = int(match.groups()[2])
                        word = match.groups()[-1]

                        ##################################
                        ##    Extract Features Here !   ##
                        ##################################

                        if word == "*pro*":
                            is_azp = False
                            if not first:
                                is_azp = True
                                azps.append((sentence_index,begin_word_index,antecedents,is_azp))

                        '''
                        if word == "*pro*" and (not first):
                            #print file_name,inline,res_info
                            print >> sys.stderr, file_name,inline,res_info
                            #print sentence_index,last_index
                            if (sentence_index - last_index) <= MAX:
                                #print sentence_index,last_index
                                if len(antecedents) >= 1:
                                    si,bi,ei = antecedents[-1]
                                    if (bi,ei) in candi[si]:
                                        print bi,ei
                        '''
                        if not word == "*pro*":
                            first = False
                            res_info = inline
                            last_index = sentence_index
                            antecedents.append((sentence_index,begin_word_index,end_word_index))
        
        if not inline:
            break
    return zps,azps,candi,nodes_info


def get_info_from_file(file_name,MAX=2):

    pattern = re.compile("(\d+?)\ +(.+?)$")
    pattern_zp = re.compile("(\d+?)\.(\d+?)\-(\d+?)\ +(.+?)$")

    total = 0

    inline = "new"
    f = open(file_name)
    
    sentence_num = 0

    '''
    ################################################################################
    # nodes_info: (dict) 存放着对应sentence_index下的每个sentence的 nl 和 wl #
    #    ------------- nodes_info[sentence_index] = (nl,wl)                   #
    # candi: (dict) 存放着sentence_index下的每个candidate                          #
    #    ------------- candi[sentence_index] = list of (begin_index,end_index)      #
    # zps:  (list)  存放着对应file下的每个zp                                       #
    #    ------------- item : (sentence_index,zp_index)
    # azps:  (list)  存放着对应file下的每个azp                                       #
    #    ------------- 每个item 对应着 (sentence_index,zp_index,antecedents=[],is_azp)
    #   -------------  antecedents - (sentence_index,begin_word_index,end_word_index)
    ################################################################################
    '''
    nodes_info = {}   
    candi = {}
    zps = []
    azps = []

    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()

        if line == "Leaves:":
            while True:
                inline = f.readline()
                if inline.strip() == "":break
                inline = inline.strip()
                match = pattern.match(inline)
                if match:
                    word = match.groups()[1]
                    #if word == "*pro*":
                    #    print word
                    #if word.find("*") < 0:
                    #    print word
            sentence_num += 1
    
        elif line == "Tree:":
            candi[sentence_num] = []
            nodes_info[sentence_num] = None
            parse_info = ""
            inline = f.readline()
            while True:
                inline = f.readline()
                if inline.strip("\n") == "":break
                parse_info = parse_info + " " + inline.strip()    
            parse_info = parse_info.strip()            
            nl,wl = parse_analysis.buildTree(parse_info)

            nodes_info[sentence_num] = (nl,wl)

            for node in nl:
                if node.tag.find("NP") >= 0:
                    if node.parent.tag.find("NP") >= 0:
                        if not (node == node.parent.child[0]):
                            continue
                    leaf_nodes = node.get_leaf()
                    if is_pro(leaf_nodes):
                        continue

                    candi[sentence_num].append((leaf_nodes[0].index,leaf_nodes[-1].index))
                    total += 1
            for node in wl:
                if node.word == "*pro*":
                    zps.append((sentence_num,node.index))  
 
        elif line.startswith("Coreference chain"):
            first = True
            res_info = None
            last_index = 0
            antecedents = []

            while True:
                inline = f.readline()
                if not inline:break
                if inline.startswith("----------------------------------------------------------------------------------"):
                    break
                inline = inline.strip()
                if len(inline) <= 0:continue
                if inline.startswith("Chain"):
                    first = True
                    res_info = None
                    last_index = 0
                    antecedents = []
                else:
                    match = pattern_zp.match(inline)
                    if match:
                        sentence_index = int(match.groups()[0])
                        begin_word_index = int(match.groups()[1])
                        end_word_index = int(match.groups()[2])
                        word = match.groups()[-1]

                        ##################################
                        ##    Extract Features Here !   ##
                        ##################################

                        if word == "*pro*":
                            is_azp = False
                            if not first:
                                is_azp = True
                                azps.append((sentence_index,begin_word_index,antecedents,is_azp))

                        '''
                        if word == "*pro*" and (not first):
                            #print file_name,inline,res_info
                            print >> sys.stderr, file_name,inline,res_info
                            #print sentence_index,last_index
                            if (sentence_index - last_index) <= MAX:
                                #print sentence_index,last_index
                                if len(antecedents) >= 1:
                                    si,bi,ei = antecedents[-1]
                                    if (bi,ei) in candi[si]:
                                        print bi,ei
                        '''
                        if not word == "*pro*":
                            first = False
                            res_info = inline
                            last_index = sentence_index
                            antecedents.append((sentence_index,begin_word_index,end_word_index))
        
        if not inline:
            break
    return zps,azps,candi,nodes_info
def main():
    path = sys.argv[1]
    paths = get_dir.get_all_file(path,[])
    for p in paths:
        if p.strip().endswith("DS_Store"):continue
        file_name = p.strip()
        if file_name.endswith("onf"):
            print >> sys.stderr, "Read File : %s"%file_name
            zps,candi,nodes_info = get_info_from_file(file_name,2)
            for (sentence_index,begin_word_index,antecedents,is_azp) in zps:
                if is_azp:
                    nl,wl = nodes_info[sentence_index]
                    print wl[begin_word_index].word
if __name__ == "__main__":
    main()
