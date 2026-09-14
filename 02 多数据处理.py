from collections import Counter
from pandas import DataFrame

import jieba
import logging
import os
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

    @classmethod
    def open_files(cls, file_path):
        """ 检查文件内容 """
        with open(file_path, "r", encoding="GBK") as file:
            """ 由于文件之前在 win 上使用过，所以文件编码为 GBK """
            email_content = file.read()
            # print(f"邮件内容：\n{email_content}")  # DEBUG

            return email_content

    @classmethod
    def cut_and_wash_words(cls, contents, stop_word):
        """ 邮件内容分词 """
        # 构建分词列表（最终形成二维列表）
        cut_words = []

        # 接受列表形式内容
        for content in contents:
            # print(content)  # DEBUG
            # jieba 分词
            words = list(jieba.cut(content, cut_all=False))
            # 去除停用词
            cut_words.append([word for word in words if word not in stop_word])

        # 打印分词列表
        # print(f"分词列表长度{len(cut_words)}")  # DEBUG
        # print(f"分词结果：{cut_words}")  # DEBUG

        return cut_words

    @classmethod
    def get_words_frequency(cls, words_list):
        """ 获取词频 """
        # 构建词频统计列表
        word_frequency = []

        # 构建分词列表
        vocabularies = []

        # 由于传入的列表是二维的，所以需要展开
        for words in words_list:
            # print(f"单词列表长度：{len(words)}")  # DEBUG
            # print(f"单词列表内容：{words}")  # DEBUG

            # 词频统计
            word_freq = dict(Counter(words))  # 转化为字典形式
            # print(f"词频统计结果：{word_freq}")   # DEBUG
            word_frequency.append(word_freq)

        # 打印词频统计列表
        # print(f"词频统计列表长度：{len(word_frequency)}")
        # print(f"词频统计列表内容：\n{word_frequency}")

        return word_frequency


class BayesTraverseFiles(object):
    """ 朴素贝叶斯分类器遍历文件夹处理类 """

    @classmethod
    def traverse_files(cls, filepath):
        """ 遍历文件夹 """
        # 获取目录中的所有文件
        files = os.listdir(filepath)
        # 文件排序
        files.sort()

        # 构建文件打开内容的列表
        contents = []

        # 遍历测试数量
        test_num = 10
        # 遍历文件并打开
        for filename in files[0:test_num]:
            # print(f"正在打开文件：{filename}")  # DEBUG
            full_file_path = os.path.join(filepath, filename)
            # print(full_file_path)  # DEBUG
            # 打开文件
            content = BayesSingleDataConductor.open_files(full_file_path)

            # 过滤空邮件
            if content is not None:
                # 过滤邮件中的非中文字符
                chinese_content = re.sub(r"[^\u4e00-\u9fa5]+", "", content)
                # print(f"过滤后的邮件内容：{chinese_content}")  # DEBUG
                contents.append(chinese_content)
        # print(f"文件夹内容列表数量：{len(contents)}")  # DEBUG
        # print(f"文件夹内容列表内容：{contents}")  # DEBUG

        return contents


def main():
    """ 朴素贝叶斯邮件分类器的主函数 """
    # 不同邮件数据的地址
    normal_emails = "data/normal_7063"
    spam_emails = "data/spam_7775"
    test_emails = "data/test_392"

    # 获取停用词
    stop_words = BayesSingleDataConductor.get_stop_words()

    # 遍历正常邮件的文件夹并获取列表式内容
    contents = BayesTraverseFiles.traverse_files(normal_emails)

    # 获取并清洗分词
    cut_words_list = BayesSingleDataConductor.cut_and_wash_words(contents, stop_words)

    # 词频统计
    BayesSingleDataConductor.get_words_frequency(cut_words_list)


if __name__ == "__main__":
    main()
