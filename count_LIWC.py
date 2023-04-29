# -*- coding: utf-8 -*-
'''
Created on Nov. 11, 2014

@author: chaos

Modified on July 18, 2019

@author: osama khalid
'''
import argparse, codecs, os, sys

class LIWC_Counter:

    def load_liwc_dict(self, liwcdic_file):
        file_content = codecs.open(liwcdic_file, "r", "utf-8").read()
        cate_text = file_content[file_content.find("%")+1:file_content[1:].find("%")].strip()
        for line in cate_text.split("\r\n"):
            self.liwc_cate_name_by_number[int(line.strip().split("\t")[0])] = line.strip().split("\t")[1]

        dict_text = file_content[file_content[1:].find("%")+2:].strip()
        for line in dict_text.split("\r\n"):
            self.liwc_cate_number_by_word[line.strip().split("\t")[0]] = set([int(item) for item in line.strip().split("\t")[1:]])

    def __init__(self, liwcdic_file):
        self.liwc_category_names = ["WC","funct","pronoun","ppron","i","we","you","shehe","they","ipron","article","prep","auxverb","adverb","conj","negate","verb","adj","compare","interrog","number","quant","affect","posemo","negemo","anx","anger","sad","social","family","friend","female","male","cogproc","insight","cause","discrep","tentat","certain","differ","percept","see","hear","feel","bio","body","health","sexual","ingest","drives","affiliation","achieve","power","reward","risk","focuspast","focuspresent","focusfuture","relativ","motion","space","time","work","leisure","home","money","relig","death","informal","swear","netspeak","assent","nonflu","filler"]
        self.liwc_cate_name_by_number = {}
        self.liwc_cate_number_by_word = {}

        if os.path.exists(liwcdic_file) == False:
            print "Cannot find LIWC dict file (path: {})".format(liwcdic_file)
            sys.exit()
        else:
            self.load_liwc_dict(liwcdic_file)

    def get_count_for_all_liwc_categories(self, text):
        count_by_categories = {"WC":0,"funct":0,"pronoun":0,"ppron":0,"i":0,"we":0,"you":0,"shehe":0,"they":0,"ipron":0,"article":0,"prep":0,"auxverb":0,"adverb":0,"conj":0,"negate":0,"verb":0,"adj":0,"compare":0,"interrog":0,"number":0,"quant":0,"affect":0,"posemo":0,"negemo":0,"anx":0,"anger":0,"sad":0,"social":0,"family":0,"friend":0,"female":0,"male":0,"cogproc":0,"insight":0,"cause":0,"discrep":0,"tentat":0,"certain":0,"differ":0,"percept":0,"see":0,"hear":0,"feel":0,"bio":0,"body":0,"health":0,"sexual":0,"ingest":0,"drives":0,"affiliation":0,"achieve":0,"power":0,"reward":0,"risk":0,"focuspast":0,"focuspresent":0,"focusfuture":0,"relativ":0,"motion":0,"space":0,"time":0,"work":0,"leisure":0,"home":0,"money":0,"relig":0,"death":0,"informal":0,"swear":0,"netspeak":0,"assent":0,"nonflu":0,"filler":0}

        count_by_categories["WC"] = len(text.split())

        for word in text.split():

            cate_numbers_word_belongs = set([])
            if self.liwc_cate_number_by_word.has_key(word):
                cate_numbers_word_belongs = self.liwc_cate_number_by_word[word]

            else:

                #liwc words have *. eg: balcon*
                word = word[:-1]
                while len(word) > 0:
                    if self.liwc_cate_number_by_word.has_key(word+"*"):
                        cate_numbers_word_belongs = self.liwc_cate_number_by_word[word+"*"]
                        break
                    else:
                        word = word[:-1]

            for num in cate_numbers_word_belongs:
                count_by_categories[self.liwc_cate_name_by_number[num]] += 1

        return count_by_categories


if (__name__ == "__main__"):
    ################################################
    #set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--liwcdic", action="store", dest="liwcdic", default="LIWC2007_English100131.dic", help="LIWC dic")
    parser.add_argument("--input", action="store", dest="input", default="input.txt", help="The input file")
    parser.add_argument("--output", action="store", dest="output", default="output.txt", help="The output file")

    args = parser.parse_args()
    liwcdic_file = args.liwcdic
    input_file = args.input
    output_file = args.output

    if output_file == "output.txt":
        output_file = input_file+".liwc_count.txt"

    ################################################
    liwc_counter = LIWC_Counter(liwcdic_file)

    fw = codecs.open(output_file, "w", "utf-8")
    
    fw.write(u"ID\t")
    fw.write(u"{0}\n".format(u"\t".join( liwc_counter.liwc_category_names )))

    
    for line in codecs.open(input_file, encoding="utf-8"):

        if line.find("\t") == -1:
            continue

        id = line[:line.find("\t")].strip()
        text = line[line.find("\t")+1:].strip()
        count_by_categories = liwc_counter.get_count_for_all_liwc_categories(text)

        fw.write(u"{0}".format(id))
        for cate in liwc_counter.liwc_category_names:
            fw.write(u"\t{0}".format(count_by_categories[cate]))
        fw.write(u"\n")
                    
    fw.close() 
            
        
        
        
    
    
    
    