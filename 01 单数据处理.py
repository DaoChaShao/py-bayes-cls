"""
文本分类的步骤：
1.文本表示
2.特征提取
3.分类器设计
4.文本分类性能评测

本次实验使用的是朴素贝叶斯分类器。

朴素贝叶斯分类器

垃圾邮件分类

数据集来源：
https://github.com/yingzk/MyML/tree/master/C-SpamClassifier/SpamClassifier/data
"""

from collections import Counter
from pandas import DataFrame

import jieba
import logging
import re
import pandas as pd

jieba.setLogLevel(logging.CRITICAL)


class BayesSingleDataConductor(object):
    """ 朴素贝叶斯分类器单数据集处理类 """

    @staticmethod
    def get_stop_words():
        """ 获取停用词 """
        # 停用词文件路径
        stop_words_filepath = "data/stop_words.txt"
        # 打开停用词文件
        with open(stop_words_filepath, "r", encoding="GBK") as file:
            # 将停用词添加至列表
            stop_words = [word.strip() for word in file.readlines()]
            # print(f"停用词列表长度：{len(stop_words)}")  # DEBUG
            # print(f"停用词列表内容：{stop_words}")  # DEBUG

            return stop_words

    @staticmethod
    def open_files():
        """ 检查文件内容 """
        with open("data/normal_7063/202", "r", encoding="GBK") as f:
            """ 由于文件之前在 win 上使用过，所以文件编码为 GBK """
            email_content = f.read()
            # print(f"邮件内容：\n{email_content}")  # DEBUG

            return email_content

    @classmethod
    def cut_and_wash_words(cls, content, stop_word):
        """ 邮件内容分词 """
        # 过滤空邮件
        if content is not None:
            # 过滤邮件中的非中文字符
            chinese_content = re.sub(r"[^\u4e00-\u9fa5]+", "", content)
            # print(f"过滤后的邮件内容：{chinese_content}")  # DEBUG

            # jieba 分词
            words = list(jieba.cut(chinese_content, cut_all=False))
            for word in words:
                # 去除停用词
                if word in stop_word:
                    words.remove(word)

            # print(f"分词结果：{words}")  # DEBUG

            return words

    @classmethod
    def get_words_frequency(cls, words_list):
        """ 获取词频 """
        # 词频统计
        word_freq = dict(Counter(words_list))  # 转化为字典形式

        # 构建 DataFrame
        cols = ["词语", "词频"]
        df = DataFrame(data=word_freq.items(), columns=cols)
        df.sort_values(by="词频", ascending=False, inplace=True)  # 按词频降序排列
        pd.set_option("display.max_rows", None)
        print(f"词频统计结果：")  # DEBUG
        print(df)  # DEBUG
        print(f"DataFrame Shape: {df.shape}")  # DEBUG

        return word_freq


def main():
    """ 朴素贝叶斯邮件分类器的主函数 """
    # 获取停用词
    stop_words = BayesSingleDataConductor.get_stop_words()

    # 打开文件
    contents = BayesSingleDataConductor.open_files()

    # 获取并清洗分词
    cut_words_list = BayesSingleDataConductor.cut_and_wash_words(contents, stop_words)

    # 词频统计
    BayesSingleDataConductor.get_words_frequency(cut_words_list)


if __name__ == "__main__":
    main()
