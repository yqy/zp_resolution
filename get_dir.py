#!/usr/bin/env python
#coding=utf-8
import os
import sys

def get_all_file(path,dir_list):
    "delete all folers and files"
    if os.path.isfile(path):
        dir_list.append(path)
        #os.remove(path)
    elif os.path.isdir(path):
        for item in os.listdir(path):
            itemsrc = os.path.join(path, item)
            get_all_file(itemsrc,dir_list)
    return dir_list

if __name__ == "__main__":
    dirname = sys.argv[1].strip()
    print "\n".join(get_all_file(dirname,[]))
