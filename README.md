**2021年3月3日星期三学习报告：**

这周学习了命名实体识别任务中关于`BILSTM+CRF`的原理与代码实现，文本数据预处理（提取识别类别，实体标记编码转换，文本分割，长短句处理，提取词性与词边界，获取拼音特征等）

具体如下：

> **BILSTM+CRF**

将输入的句子先进行embedding处理，作为LSTM（长短时记忆神经网络）的输入，输出并没有采用常见的全连接softmax处理，因为这样并不能将词性之间的依赖性表现出来，（举个例子，若采用BIO标注，‘B’后面是不能连接‘O’的），本论文中使用的CRF（链式条件随机场）来做进一步的处理，使用了‘动态规划’的思想，节省了时间和空间的消耗。在选择最佳输出结果时，采用维特比算法，快速地找到了输出结果。

 

> **文本数据预处理：**

从论文中扫描出文字，接着对这些非结构化数据进行标注，使用的特征标注有：[word, flag, bound, label, radical, pinyin]

初始文本：

![文本, 信件  描述已自动生成](file:///C:/Users/CHRISI~1/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png)

 

label（根据人工给定的实体标记）：

![img](file:///C:/Users/CHRISI~1/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png)

 

标注特征结果：[word, bound（BMES标注方式）, flag（结巴分词中的词性标注）, label（根据人工给定的实体标记）, radical（偏旁部首）, pinyin]

 

![文本, 信件  描述已自动生成](file:///C:/Users/CHRISI~1/AppData/Local/Temp/msohtmlclip1/01/clip_image006.png)

 