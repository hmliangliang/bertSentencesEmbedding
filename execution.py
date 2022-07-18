#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:38
# @Author  : Liangliang
# @File    : execution.py
# @Software: PyCharm

import jieba
import pandas as pd
import s3fs
import time
import math
import numpy as np
import argparse
import datetime
import os
import base64
from multiprocessing.dummy import Pool
os.system("pip install dataclasses")
os.system("pip install gensim")
os.system("pip install tarfile")
from gensim.models import KeyedVectors
import requests
import tarfile
result = 0
corpus = 0

def multiprocessingWrite(file_number,data,output_path,count):
    #print("开始写第{}个文件 {}".format(file_number,datetime.datetime.now()))
    n = len(data)  # 列表的长度
    #s3fs.S3FileSystem = S3FileSystemPatched
    #fs = s3fs.S3FileSystem()
    with open(os.path.join(output_path, 'pred_{}_{}.csv'.format(count,int(file_number))), mode="a") as resultfile:
        if n > 1:#说明此时的data是[[],[],...]的二级list形式
            for i in range(n):
                line = ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
        else:#说明此时的data是[x,x,...]的list形式
            line = ",".join(map(str, data)) + "\n"
            resultfile.write(line)
    print("第{}个大数据文件的第{}个子文件已经写入完成,写入数据的行数{} {}".format(count,file_number,n,datetime.datetime.now()))

class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


class S3Filewrite:
    def __init__(self, args):
        super(S3Filewrite, self).__init__()
        self.output_path = args.data_output


def write(data, args, count):
    #注意在此业务中data是一个二维list
    n_data = len(data) #数据的数量
    n = math.ceil(n_data/args.file_max_num) #列表的长度
    start = time.time()
    for i in range(0,n):
        multiprocessingWrite(i, data[i * args.file_max_num:min((i + 1) * args.file_max_num, n_data)],
                                 args.data_output, count)
    cost = time.time() - start
    print("write is finish. write {} lines with {:.2f}s".format(n_data, cost))

def getEmbedding(sentence, i, roleid, args, count):
    '''sentences:分词的句子,为base64编码，需要先解码
    i:序号
    roleid: 数据的id
    '''
    sentences = str(base64.b64decode(sentence), 'utf-8')
    global result
    if i%100000 == 0:
        print("第{}个文件的第{}个任务开始执行! {}".format(count, i, datetime.datetime.now()))
    flag = False #表示当前的sentences的分词结果是否都可以查到词向量 False: 表示所有的词都没有词向量  True:表示至少有一个词可以查到词向量
    value = np.zeros((1, args.dim))
    if len(sentences) > 0:
        words = jieba.cut(sentences)
        counts = 0
        for word in words:
            try:
                value = value + corpus[word]
                counts = counts + 1
                flag = True
            except KeyError:
                pass
    else:
        flag = False
    if flag == True:#有句子向量
        value = value/counts
        value = value.astype("str")
        result[i, 0] = str(roleid)
        result[i, 1] = sentence
        result[i, 2::] = value
    else:#无句子向量输出
        result[i, 0] = str(roleid)
        result[i, 1] = sentence
        result[i, 2::] = "no"
    if i%100000 == 0:
        print("第{}个文件的第{}个任务执行完成! {}".format(count, i, datetime.datetime.now()))

if __name__ == "__main__":
    #配置参数
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--dim", help="单词表中词embedding的维数", type=int, default=100)
    parser.add_argument("--thread_num", help="多线程编程的线程数目", type=int, default=1000)
    parser.add_argument("--file_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=5000000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    print("开始下载word embedding文件! {}".format(datetime.datetime.now()))
    #远程下载文件  https://zhuanlan.zhihu.com/p/106309634
    url = 'https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d100-v0.2.0.tar.gz'
    res = requests.get(url, stream=True)
    total_length = int(res.headers.get('content-length'))
    with open("tencent-ailab-embedding-zh-d100-v0.2.0.tar.gz", "wb") as pypkg:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:
                pypkg.write(chunk)
    print("词嵌入文件下载完成, 开始解压下载文件! {}".format(datetime.datetime.now()))
    #解压文件
    tar = tarfile.open("tencent-ailab-embedding-zh-d100-v0.2.0.tar.gz")
    tar.extractall(path="tencent-ailab-embedding-zh-d100-v0.2.0")
    tar.close()
    print("词嵌入文件解压完成! {}".format(datetime.datetime.now()))
    #删除已下载的tar.gz文件,节省存储空间
    os.remove("tencent-ailab-embedding-zh-d100-v0.2.0.tar.gz")
    #打印文件首行看文件是否正常
    f = open("./tencent-ailab-embedding-zh-d100-v0.2.0/tencent-ailab-embedding-zh-d100-v0.2.0/tencent-ailab-embedding-zh-d100-v0.2.0.txt",'r',encoding="utf8")
    print("词嵌入文件的首行信息:",f.readline())
    f.close()

    # 读取数据文件
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    corpus = KeyedVectors.load_word2vec_format('./tencent-ailab-embedding-zh-d100-v0.2.0/tencent-ailab-embedding-zh-d100-v0.2.0/tencent-ailab-embedding-zh-d100-v0.2.0.txt', binary=False,
                                               encoding="utf8")
    #删除txt文件节省空间内存11G
    os.remove("./tencent-ailab-embedding-zh-d100-v0.2.0/tencent-ailab-embedding-zh-d100-v0.2.0/tencent-ailab-embedding-zh-d100-v0.2.0.txt")
    #处理数据
    for file in input_files:
        pool = Pool(processes=args.thread_num)
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        data = pd.read_csv("s3://" + file, sep=',', header=None, usecols=[0,1]).astype('str')  # 读取数据,第一列为id,第二列为中文txt
        n = data.shape[0]
        result = np.zeros((n, args.dim + 2)).astype("str")
        for i in range(n):
            pool.apply_async(func=getEmbedding, args=(data.iloc[i, 1], i, data.iloc[i, 0], args, count,))
        pool.close()
        pool.join()
        write(result.tolist(), args, count)
    print("已完成第{}个文件数据的推断! {}".format(count, datetime.datetime.now()))