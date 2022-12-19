import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
from tqdm import tnrange
from urllib.request import urlopen
import re
import requests
import urllib.request
from tqdm import tqdm
import pickle
from selenium.common.exceptions import *

driver = webdriver.Chrome(executable_path="chromedriver_path/chromedriver")
driver.maximize_window()

train = pd.read_csv("open/train.csv")

trans_list = []
backtrans_list = []

# kor_to_trans 함수 -> 한국어를 외국어로 바꾸는 함수
# trans_to_kor 함수 -> 외국어를 한국어로 바꾸는 함수
# 첫번째 함수를 먼저 수행하면 trans_list에 외국어로 번역된 텍스트가 들어있음
# trans_list를 두번째 함수의 입력값으로 받아서 다시 한글 텍스트의 리스트로 뽑아냄

sentences = train["문장"].values

backtrans_dict = {}
for i in range(len(sentences)):
    backtrans_dict[sentences[i]] = ""

def full_trans(text_data, trans_lang, save_every=100):
    for i in tqdm(range(2600, len(text_data))):
        try:
            try:
                driver.get('https://papago.naver.com/?sk=ko&tk='+trans_lang+'&st='+text_data[i])
                time.sleep(2.5)
                trans_text = driver.find_element_by_xpath('//*[@id="txtTarget"]').text
            except NoSuchElementException:
                driver.get('https://papago.naver.com/?sk=ko&tk='+trans_lang)
                driver.find_element_by_xpath('//*[@id="txtSource"]').send_keys(text_data[i])
                time.sleep(2.5)
                trans_text = driver.find_element_by_xpath('//*[@id="txtTarget"]').text
            try:
                driver.get('https://papago.naver.com/?sk='+trans_lang+'&tk=ko&st='+trans_text)
                time.sleep(2.5)
                backtrans = driver.find_element_by_xpath('//*[@id="txtTarget"]').text
                backtrans_dict[text_data[i]] = backtrans
            except NoSuchElementException:
                driver.get('https://papago.naver.com/?sk='+trans_lang+'&tk=ko')
                driver.find_element_by_xpath('//*[@id="txtSource"]').send_keys(trans_text)
                time.sleep(2.5)
                backtrans = driver.find_element_by_xpath('//*[@id="txtTarget"]').text
                backtrans_dict[text_data[i]] = backtrans

            if i%save_every == 0 and i != 0:
                print("saving...")
                with open("backtranslated_eng_2600_.pkl", "wb") as f:
                    pickle.dump(backtrans_dict, f)
                print("done!")
        except Exception as e:
            print(e)
            continue


full_trans(sentences, "en")
