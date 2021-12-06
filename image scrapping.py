#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all the required libraries
import pandas as pd
from bs4 import BeautifulSoup
import requests
import selenium
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


# In[3]:


driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://www.amazon.com")
page = requests.get("https://www.amazon.com")
soup=BeautifulSoup(page.content,'html.parser')


# In[4]:


cat=["saree for women","Trousers for men","Jeans for men"]
ims=[]
labels=[]

for z in cat:
    search_bar = driver. find_element_by_id("twotabsearchtextbox")
    search_bar.clear()
    search_bar.send_keys(z)
    search_btn = driver.find_element_by_xpath("//div[@class = 'nav-search-submit nav-sprite']/span/input")
    search_btn.click()
    for j in range(0,6):
        imgs=driver.find_elements_by_xpath("//div/img[@class='s-image']")
        for i in imgs:
            img_src = i.get_attribute("src")
            ims.append(img_src)
            if z == "saree for women" :
                labels.append("saree")
            elif z == "Trousers for men" :
                labels.append("Trouser")
            else:
                labels.append("jeans")
        nxt=driver.find_element_by_xpath("//li[@class='a-last']//a")
        nt=nxt.get_attribute("href")
        driver.get(nt)
len(ims)


# In[5]:


ims


# In[6]:


df=pd.DataFrame({'image_link':ims,
                'labels':labels})


# In[7]:


df


# In[8]:


df.to_csv('C:/Users/Poonam/image_link.csv')


# In[9]:


df=pd.read_csv("image_link.csv")
length=len(df['image_link'])


# In[10]:


import io
import hashlib
import pathlib
from PIL import Image
import numpy as np
import pickle
from pathlib import Path
import requests


# In[11]:


def getsaveimg(im,la,output_dir):
    response= requests.get(im,headers={"User-agent": "Google Chrome/90.0"})
    img_content = response.content
    img_file = io.BytesIO(img_content)
    image = Image.open(img_file).convert("RGB")
    filename =  (la) + "." + hashlib.sha1(img_content).hexdigest()[:10] + ".png"
    file_path = output_dir / filename
    image.save(file_path, "PNG", quality=80)
    imagename.append(filename)

    
imagename=[]
for i in range(0,length):
    im=df.image_link[i]
    la=df.labels[i]
    getsaveimg(im,la, output_dir = pathlib.Path("C:/Users/Poonam/imagedata"))


# In[12]:


driver.close()


# In[13]:


df.to_csv('C:/Users/Poonam/imagedata/imagename.csv')

