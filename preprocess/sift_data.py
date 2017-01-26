#coding:utf-8
try: 
  import xml.etree.cElementTree as ET 
except ImportError: 
  import xml.etree.ElementTree as ET 
import sys 
reload(sys)
sys.setdefaultencoding('utf-8')

try: 
  tree = ET.parse("SemEval2016-Task3-CQA-QL-train-part1.xml")     #打开xml文档 
  root = tree.getroot()         #获得root节点  
except Exception, e: 
  print "Error:cannot parse file:country.xml."
  sys.exit(1) 

for Que in root.findall('OrgQuestion'): #找到root节点下的所有country节点 
  for Thread in Que.findall('Thread'):
    RelQue_text = Thread.find('RelQuestion').find('RelQBody').text
    #print Que.find('OrgQBody').text,"ans: ",RelQue_text," lable: ",Thread.get('RELQ_RELEVANCE2ORGQ') 
    for Comment in Thread.findall('RelComment'):
      print RelQue_text,"hrbans: ",Comment.find('RelCText').text," lable: ",Comment.get('RELC_RELEVANCE2RELQ')
