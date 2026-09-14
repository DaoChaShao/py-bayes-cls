from collections import Counter
from pandas import DataFrame

import os
import pandas as pd
import random
import re
import thulac
import tqdm

from utils import timer


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
        # 加载并添加自定义词库
        # BayesSingleDataConductor.load_add_words()

        # 构建分词列表（最终形成二位列表）
        cut_words = []

        # 加载 thulac 分词器
        thu = thulac.thulac(
            seg_only=True,  # 只进行分词
            user_dict="data/add_words_thulac.txt",  # 自定义词库
        )
        # 切词

        # 接受列表形式内容
        for content in tqdm.tqdm(contents, desc="邮件分词中：", colour="yellow"):
            # for content in contents:
            # THULAC 分词
            words = thu.cut(
                content,
                text=True  # 输出为文本
            )
            # 去除停用词
            cut_words.append([word for word in words if word not in stop_word])

        # 打印分词列表
        # print(f"分词列表长度{len(cut_words)}")  # DEBUG
        # print(f"分词结果：{cut_words}")  # DEBUG

        return cut_words

    @classmethod
    def get_words_frequency(cls, words_list, top_num):
        """ 获取词频 """
        # 构建分词列表
        words_frequency = []

        # 由于传入的列表是二维的，所以需要展开
        for words in tqdm.tqdm(words_list, desc="词频统计中：", colour="blue"):
            # 词频统计
            words_freq = Counter(words)
            # print(f"词频统计结果：{words_freq}")   # DEBUG
            word_frequency = words_freq.most_common(top_num)  # 取词频最高的前 20 个词
            # print(f"词频最高的前 20 个词：{word_frequency}")
            words_frequency.append(word_frequency)

        # print(f"vocabularies len: {len(words_frequency)}")
        # print(f"vocabularies: {words_frequency}")

        return words_frequency

    @staticmethod
    def words_frequency_for_together_and_top(words_separated_list, top_num):
        """ 训练集 TOP 词频统计 """
        # 初始化 Counter
        words_counter = Counter()

        # 遍历邮件分词列表
        for words in words_separated_list:
            words_counter.update(dict(words))

        # 取词频最高的前 top_num 个词
        tup_words_freq = dict(words_counter.most_common(top_num))

        return tup_words_freq


class BayesTraverseFiles(object):
    """ 朴素贝叶斯分类器遍历文件夹处理类 """

    @classmethod
    def traverse_files(cls, filepath, num):
        """ 遍历文件夹 """
        # 获取目录中的所有文件
        files = os.listdir(filepath)
        # 文件排序
        files.sort()

        # 构建文件打开内容的列表
        files_contents = []

        # 遍历文件并打开
        for filename in tqdm.tqdm(files[0:num], desc="文件遍历中："):
            # for filename in files[0:num]:
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
                files_contents.append(chinese_content)
        # print(f"文件夹内容列表数量：{len(files_contents)}")  # DEBUG
        # print(f"文件夹内容列表内容：{files_contents}")  # DEBUG

        return files_contents, len(files_contents)

    @classmethod
    def traverse_test_files(cls, test_filepath):
        """ 打开测试文件 """
        # 获取目录中的所有文件
        test_files = os.listdir(test_filepath)

        # 随机选择一个文件夹
        random_test_files = random.choice(test_files)
        # print(f"随机选择的文件：{random_test_files}")  # DEBUG

        # 打开测试文件
        full_test_file_path = os.path.join(test_filepath, random_test_files)
        # print(full_test_file_path)  # DEBUG

        # 打开目标文件
        content = BayesSingleDataConductor.open_files(full_test_file_path)

        # 构建文件打开内容的列表
        contents = []

        # 过滤空邮件
        if content is not None:
            # 过滤邮件中的非中文字符
            chinese_content = re.sub(r"[^\u4e00-\u9fa5]+", "", content)
            # print(f"过滤后的邮件内容：{chinese_content}")  # DEBUG
            contents.append(chinese_content)
        # print(f"文件夹内容列表数量：{len(contents)}")  # DEBUG
        # print(f"文件夹内容列表内容：{contents}")  # DEBUG

        return contents, random_test_files


class BayesAlgorithm(object):
    """ 朴素贝叶斯分类器算法类 """

    @classmethod
    def get_test_words_prob(cls, test_freq_list, nor_freq_dict, nor_len, spam_freq_dict, spam_len):
        """ 计算测试邮件词频的“平均”概率 """
        # 构建平均概率字典的列表
        words_prob_list = []

        # 遍历测试邮件词频列表
        for test_freq in test_freq_list:
            # 将每封邮件的词频转换为字典
            test_freq = dict(test_freq)
            # 构建测试邮件词的平均概率的字典
            test_words_prob = {}

            # 遍历测试邮件词频字典
            for word in test_freq.keys():
                if word in nor_freq_dict.keys() and word in spam_freq_dict.keys():
                    # 计算 top词 在正常邮件和垃圾邮件中的平均概率
                    pw_n = nor_freq_dict[word] / nor_len
                    pw_s = spam_freq_dict[word] / spam_len
                    # 利用贝叶斯公式计算后验概率
                    ps_w = pw_s / (pw_s + pw_n)
                    # 保存到测试邮件词频字典
                    test_words_prob[word] = ps_w
                elif word in nor_freq_dict.keys() and word not in spam_freq_dict.keys():
                    pw_n = nor_freq_dict[word] / nor_len
                    pw_s = 0.01
                    ps_w = pw_s / (pw_s + pw_n)
                    test_words_prob[word] = ps_w
                elif word not in nor_freq_dict.keys() and word in spam_freq_dict.keys():
                    pw_n = 0.01
                    pw_s = spam_freq_dict[word] / spam_len
                    ps_w = pw_s / (pw_s + pw_n)
                    test_words_prob[word] = ps_w
                elif word not in nor_freq_dict.keys() and word not in spam_freq_dict.keys():
                    test_words_prob[word] = 0.5
            words_prob_list.append(test_words_prob)

        # 打印平均概率字典
        # print(f"测试邮件词频平均概率字典的列表：{words_prob_list}")  # DEBUG

        return words_prob_list

    @staticmethod
    def bayes_calculation(test_words_prob_list):
        """ 通过平均概率计算测试邮件词频的联合概率 """
        # 构建 p_value 列表
        p_values_list = []

        # 遍历平均概率的字典列表
        for test_words_prob in test_words_prob_list:
            # 计算测试邮件的联合概率
            # 先验概率
            ps_w = 1
            ps_n = 1
            # 遍历平均概率的频率
            for prob in test_words_prob.values():
                ps_w *= prob  # p1*p2*p3...pn
                ps_n *= (1 - prob)  # (1-p1)*(1-p2)*(1-p3)...(1-pn)
            # 利用贝叶斯公式计算联合概率
            p_value = float(ps_w / (ps_w + ps_n))
            # 打印后验概率
            # print(f"联合概率：{p_value}")
            p_values_list.append(p_value)

        return p_values_list


class EmailsJudgement(object):
    """ 邮件判别类 """

    @staticmethod
    def emails_evaluation(p_values, file_names):
        """ 邮件判别 """
        judgement_list = []
        for p_value, file_name in zip(p_values, file_names):
            # 判别邮件是否为垃圾邮件
            if p_value > 0.9:
                # 垃圾邮件
                judgement_list.append(1)
                # print("*" * 50)
                # print(f"名称为 {file_name} 的邮件属于 xxx 垃圾邮件 xxx")
                # print("*" * 50)
            else:
                # 正常邮件
                judgement_list.append(0)
                # print("*" * 50)
                # print(f"名称为 {file_name} 的邮件属于 *** 正常邮件 ***")
                # print("*" * 50)

        # print(f"邮件判别结果列表：{judgement_list}")
        return judgement_list


def email_content_check(p_value, file_name):
    """ 邮件内容复查 """
    if p_value > 0.9:
        with open(f"data/test_392/{file_name}", "r", encoding="GBK") as file:
            email_content = file.read()
            # 过滤邮件中的非中文字符
            chinese_content = re.sub(r"[^\u4e00-\u9fa5]+", "", email_content)
            if chinese_content is not None:
                # 打印邮件内容
                print(f"名称为 {file_name} 的邮件内容：\n{chinese_content}")
            else:
                print(f"名称为 {file_name} 的邮件内容为空")
    else:
        with open(f"data/test_392/{file_name}", "r", encoding="GBK") as file:
            email_content = file.read()
            # 过滤邮件中的非中文字符
            chinese_content = re.sub(r"[^\u4e00-\u9fa5]+", "", email_content)
            if chinese_content is not None:
                # 打印邮件内容
                print(f"名称为 {file_name} 的邮件内容：\n{chinese_content}")
            else:
                print(f"名称为 {file_name} 的邮件内容为空")


@timer
def main():
    """ 朴素贝叶斯邮件分类器的主函数 """
    # 不同邮件数据的地址
    normal_emails = "data/normal_7063"
    spam_emails = "data/spam_7775"
    test_emails = "data/test_392"

    # 获取停用词
    stop_words = BayesSingleDataConductor.get_stop_words()

    # 控制测试邮件数据数量
    # test_num = 10  # 控制测试邮件数据数量
    test_num = None  # 控制测试邮件数据数量，None 表示全部测试

    # 选取词频最高的前多少个词
    words_freq_top_num = 20

    # 统计正常邮件词频
    contents_nor, nor_len = BayesTraverseFiles.traverse_files(normal_emails, test_num)
    # print(f"正常邮件数量：{nor_len}")  # DEBUG
    cut_words_list = BayesSingleDataConductor.cut_and_wash_words(contents_nor, stop_words)
    separated_freq_nor = BayesSingleDataConductor.get_words_frequency(cut_words_list, words_freq_top_num)
    freq_nor = BayesSingleDataConductor.words_frequency_for_together_and_top(separated_freq_nor, words_freq_top_num)

    # print(f"正常邮件词频统计结果：{freq_nor}")  # DEBUG

    # 统计垃圾邮件词频
    contents_spam, spam_len = BayesTraverseFiles.traverse_files(spam_emails, test_num)
    # print(f"垃圾邮件数量：{spam_len}")  # DEBUG
    cut_words_list = BayesSingleDataConductor.cut_and_wash_words(contents_spam, stop_words)
    separated_freq_spam = BayesSingleDataConductor.get_words_frequency(cut_words_list, words_freq_top_num)
    freq_spam = BayesSingleDataConductor.words_frequency_for_together_and_top(separated_freq_spam, words_freq_top_num)
    # print(f"垃圾邮件词频统计结果：{freq_spam}")  # DEBUG

    # 构建测试集数据内容列表
    test_data_contents_list = []
    # 构建测试集数据名称列表
    test_data_names_list = []
    # 获取测试邮件目录中的所有文件
    test_data_files = os.listdir(test_emails)
    # 遍历测试邮件文件
    for test_data_file_mane in test_data_files[:]:
        # 合成邮件文件路径
        test_data_full_file_path = os.path.join(test_emails, test_data_file_mane)
        # 打开邮件文件
        test_data_contents = BayesSingleDataConductor.open_files(test_data_full_file_path)
        if test_data_contents is not None:
            # 过滤邮件中的非中文字符
            test_data_chinese_content = re.sub(r"[^\u4e00-\u9fa5]+", "", test_data_contents)
            # print(f"过滤后的邮件内容：{test_data_chinese_content}")  # DEBUG
            test_data_contents_list.append(test_data_chinese_content)
            test_data_names_list.append(test_data_file_mane)

    # print(f"test_data_contents_list len: {len(test_data_names_list)}")  # DEBUG
    # print(f"test_data_contents_list：{test_data_contents_list}")  # DEBUG

    # 邮件内容分词
    test_data_cut_words_list = BayesSingleDataConductor.cut_and_wash_words(test_data_contents_list, stop_words)
    # print(f"test_data_cut_words_list len: {len(test_data_cut_words_list)}")  # DEBUG
    # print(f"test_data_cut_words_list：{test_data_cut_words_list}")  # DEBUG

    # 统计测试邮件词频
    test_data_freq = BayesSingleDataConductor.get_words_frequency(test_data_cut_words_list, words_freq_top_num)
    # print(f"test_data_freq len: {len(test_data_freq)}")  # DEBUG
    # print(f"test_data_freq：{test_data_freq}")  # DEBUG

    # 计算测试邮件词频的平均概率
    test_data_test_words_prob = BayesAlgorithm.get_test_words_prob(test_data_freq, freq_nor, nor_len, freq_spam,
                                                                   spam_len)
    # print(f"test_data_test_words_prob len: {len(test_data_test_words_prob)}")  # DEBUG
    # print(f"test_data_test_words_prob: {test_data_test_words_prob}")  # DEBUG

    # 计算测试邮件词频的联合概率
    test_data_p_value = BayesAlgorithm.bayes_calculation(test_data_test_words_prob)
    # print(f"test_data_p_value len: {len(test_data_p_value)}")  # DEBUG
    # print(f"test_data_p_value: {test_data_p_value}")  # DEBUG

    # 测试邮件名称
    # print(f"test_data_names_list: {test_data_names_list}")  # DEBUG

    # 邮件判别
    emails_judgement_list = EmailsJudgement.emails_evaluation(test_data_p_value, test_data_names_list)

    # 统计邮件判别结果
    count_nor = emails_judgement_list.count(0)
    count_spam = emails_judgement_list.count(1)
    # 计算邮件判别结果的百分比
    percentage_nor = count_nor / len(emails_judgement_list) * 100
    percentage_spam = count_spam / len(emails_judgement_list) * 100
    # 打印邮件判别结果
    print("*" * 50)
    print(f"测试邮件共：{len(emails_judgement_list)} 封")
    print("-" * 50)
    print(f"其中，正常邮件：{count_nor} 封，占比约：{percentage_nor:.2f} %")
    print(f"其中，垃圾邮件：{count_spam} 封，占比约：{percentage_spam:.2f} %")
    print("*" * 50)


if __name__ == "__main__":
    main()
