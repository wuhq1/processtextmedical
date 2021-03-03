import os
import pandas as pd
import pickle
from collections import Counter
from tqdm import tqdm
import jieba.posseg as psg#结巴的词性标注
from cnradical import Radical, RunOption
from dataprocess import split_text
import shutil#创建一棵树的库
from random import shuffle
train_dir = 'ruijin_round1_train2_20181022'
def process_text(idx, split_method=None, split_name ='train'):
    '''
    读取文本 切割 然后打上标记 并提取边界、词性、
    偏旁部首、拼音等文本特征'''
    data ={} #一个很大的字典，包含很多内容
    # ////////////////获取句子///////////////////////////
    if split_method is None:
        with open(f'{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        with open(f'{train_dir}/{idx}.txt', 'r', encoding='utf-8') as f:
            texts = f.read()#读取整篇文章
            texts = split_method(texts)
    data['word']=texts

    # /////////////////获取标签/////////////////////////////

    tag_list = ['O' for s in texts for x in s]# 双重循环
    # print(tag_list)
    tag = pd.read_csv(f'{train_dir}/{idx}.ann', header=None, sep='\t')
    for i in range(tag.shape[0]):
        tag_item = tag.iloc[i][1].split(' ')
        cls, start, end = tag_item[0], int(tag_item[1]), int(tag_item[-1])
        tag_list[start] = 'B-'+ cls #tag_list的下标开始改动
        for j in range(start+1, end):
            tag_list[j] = 'I-'+cls
    # return tag_list# 只是弄好了一个全文章的标记，但是我们要做成一句话形式的标记匹配

    assert len([x for s in texts for x in s]) == len(tag_list) # 保证两个序列长度一致


    # text_list =''
    # for t in texts:
    #     text_list += t
    # textes = []
    # tags = []
    # start = 0
    # end = 0
    # max = len(tag_list)
    # for s in texts:
    #     l = len(s)
    #     end += l
    #     tags.append(tag_list[start:end])
    #     start +=l
    # data['label'] = tags  # 做好标签
    # # print(tags,texts) #做好了标签与文本的对应关系


    #///////////////提取词性和词边界特征/////////////////
    word_bounds =['M' for item in tag_list]#保存词语的分词边界
    word_flags = []
    for text in texts:
        for word , flag in psg.cut(text):# word 中国，flag：ns
            if len(word)==1:#说明是单个词
                start = len(word_flags)
                word_bounds[start]= 'S'
                word_flags.append(flag)
            else:
                start =len(word_flags)
                word_bounds[start]='B'
                word_flags+=[flag]*len(word)
                end =len(word_flags)-1
                word_bounds[end]='E'

    #////////统一截断///////////////

    bounds =[]
    flags =[]
    tags =[]
    start = 0
    end = 0
    for s in texts:
        l =len(s)
        end +=l
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        tags.append(tag_list[start:end])
        start +=l
    data['bound'] = bounds
    data['flag']= flags
    data['label'] = tags
    # return texts[0], tags[0], bounds[0], flags[0]#此处已经完成了以上四个特征的输出

   # /////////////////获取拼音特征和偏旁部首/////////////////////

    radical = Radical(RunOption.Radical)
    pinyin =Radical(RunOption.Pinyin)
    #对于没有偏旁部首的字标上UNK
    data['radical']=[[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in s ] for s in texts]
    data['pinyin'] =[[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK'for x in s]for s in texts]
    # return texts[0], tags[0], bounds[0], flags[0], data['radical'][0], data['pinyin'][0]
    #数据的几个特征都对应上了。

    #//////////////////////////存储数据///////////////////////////////////

    num_samples = len(texts)
    num_col = len(data.keys())
    dataset = []
    for i in range(num_samples):
        records = list(zip(*[list(v[i]) for v in data.values()]))
        #这个符号是解压的意思
        dataset+=records+[['sep']*num_col]  # 一句话结束后，要用sep进行隔开。
    dataset = dataset[:-1] #最后一个sep不要隔开
    dataset=pd.DataFrame(dataset, columns=data.keys())
    save_path = f'data/{split_name}/{idx}.csv'

    def clean_word(w): #现在可以去掉空格，标记已经打好了。
        if w =='\n':
            return 'LB'
        if w in [' ', '\t', '\u2003']:
            return 'SPACE'
        if w.isdigit():
            return 'NUM' #将数字变成统一的数字
        return w

    dataset['word']=dataset['word'].apply(clean_word)

    dataset.to_csv(save_path, index=False, encoding = 'utf-8')
# 只是处理一个文件而已,在train文件夹中生成了0.csv文件。


#///////////////////进行批量处理/////////////////////////////////////

def multi_process(split_method=None,train_ratio=0.8):
    if os.path.exists('data/prepare/'):
        shutil.rmtree('data/prepare/')#如果目录存在的话，那就删除去，用rmtree
    if not os.path.exists('data/prepare/train/'):#创建目录
        os.makedirs('data/prepare/train') #mkdir只能创建当前目录 ，不能创建三级目录
        os.makedirs('data/prepare/test')
    idxs=list(set([file.split('.')[0] for file in os.listdir(train_dir)]))#获取所有文件的名字
    shuffle(idxs)#打乱顺序
    index=int(len(idxs)*train_ratio)#拿到训练集的截止下标
    train_ids=idxs[:index]#训练集文件名集合
    test_ids=idxs[index:]#测试集文件名集合

    import multiprocessing as mp #引入多进程
    num_cpus=mp.cpu_count()#获取机器cpu的个数
    pool=mp.Pool(num_cpus)
    results=[]
    for idx in train_ids:
        result=pool.apply_async(process_text,args=(idx,split_method,'train'))# args中的值是process_text的参数
        results.append(result)
    for idx in test_ids:
        result=pool.apply_async(process_text,args=(idx,split_method,'test'))
        results.append(result)#没有返回值，就不用区别train_result和test_result
    pool.close()
    pool.join()
    [r.get() for r in results]

#统计字典
def mapping(data,threshold=10,is_word=False,sep='sep',is_label=False):
    count=Counter(data)#返回的是一个字典
    if sep is not None:
        count.pop(sep)
    if is_word:
        #先按照句子长度进行排序，再去分批次处理，这样会得到长度相似的句子
        count['PAD']=100000001 # 要填充空处
        count['UNK']=100000000
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)#降序 返回的是一个列表
        data=[x[0] for x in data if x[1]>=threshold]#去掉频率小于threshold的元素  未登录词
        id2item=data
        item2id={id2item[i]:i for i in range(len(id2item))}
    elif is_label:
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    else:
        count['PAD'] = 100000001
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    return id2item,item2id



def get_dict():# 生成映射词典
    map_dict={}
    from glob import glob#遍历文件的一个工具
    all_w,all_bound,all_flag,all_label,all_radical,all_pinyin=[],[],[],[],[],[]
    for file in glob('data/prepare/train/*.csv')+glob('data/prepare/test/*.csv'):
        df=pd.read_csv(file,sep=',')
        all_w+=df['word'].tolist()
        all_bound += df['bound'].tolist()
        all_flag += df['flag'].tolist()
        all_label += df['label'].tolist()
        all_radical += df['radical'].tolist()
        all_pinyin += df['pinyin'].tolist()
    map_dict['word']=mapping(all_w,threshold=20,is_word=True)
    map_dict['bound']=mapping(all_bound)
    map_dict['flag']=mapping(all_flag)
    map_dict['label']=mapping(all_label,is_label=True)
    map_dict['radical']=mapping(all_radical)
    map_dict['pinyin']=mapping(all_pinyin)

    with open(f'data/prepare/dict.pkl','wb') as f:#保存以上文件
        pickle.dump(map_dict,f)


if __name__ == '__main__':
    # print(process_text('0',split_method=split_text,split_name='train'))
    multi_process(split_text)
#将字的特征变成向量，组合成这一个字的特征向量



