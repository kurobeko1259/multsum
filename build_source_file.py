# -*- coding: utf-8 -*-
import os
from nltk.tokenize import sent_tokenize
import xml.etree.ElementTree as ET
import re

outputpath = "/media/suzukilab/UbuntuData/corpus/obesity/body/"
outputpath_summary = "/media/suzukilab/UbuntuData/corpus/obesity/summary/"

sentences = []

def reform_text(document):
    re_texts = []
    for text in document:
        re_text = re.sub('[^a-zA-Z0-9]+', ' ', text)
        re_text = re.sub(r"\s+", " ", re_text)
        re_texts.append(re_text)

    return re_texts



def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)
