#coding=utf8
import sys
from numpy import linalg
from numpy import array
from numpy import inner

def cosin(Al,Bl):
    #Al,Bl 向量的list
    A = array(Al)
    B = array(Bl)
    nA = linalg.norm(A)
    nB = linalg.norm(B)
    if nA == 0 or nB == 0:
        return 0
    #num = float(A * B.T) #行向量
    num = inner(A,B)
    denom = nA * nB
    cos = num / denom #余弦值
    #sim = 0.5 + 0.5 * cos #归一化
    sim = cos
    return sim


class Word2Vec:
    word_dict = {}
    def __init__(self,w2v_dir="../data/word2vec",dimention=100):
        self.dimention = dimention
        f = open(w2v_dir)
        line = f.readline()
        while True:
            line = f.readline()
            if not line:break
            line = line.strip().split(" ")
            word = line[0]
            vector = line[1:]
            vec = [float(item) for item in vector]
            self.word_dict[word] = array(vec)
    def get_vector_by_word(self,word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return array([0.0]*self.dimention)
    def get_vector_by_word_dl(self,word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return None

    def get_vector_by_list(self,wl):
        result = array([0.0]*self.dimention)
        for word in wl:
            result += self.get_vector_by_word(word)
        if len(wl) == 0:
            return array([0.0]*self.dimention)
        return result/len(wl)

def main():
    w2v = Word2Vec("../data/word2vec")
    print "go"
    while True:
        line = sys.stdin.readline()
        if not line:break
        line = line.strip().split(" ")
        l1 = w2v.get_vector_by_word(line[0].strip())
        l2 = w2v.get_vector_by_word(line[1].strip())
        #print l1
        #print l2
        print cosin(l1,l2)

def tt():
    w2v = Word2Vec()
    l1 = array(w2v.get_vector_by_list(["国王","你好"]))
    #l2 = array(w2v.get_vector("男"))
    #l3 = array(w2v.get_vector("女"))
    #l = l1 - l2 + l3
    #ll = w2v.get_vector("皇后") 
    #print cosin(ll,list(l))
    print l1

    
if __name__ == "__main__":
    main()
    #tt()
