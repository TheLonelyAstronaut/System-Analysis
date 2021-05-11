import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import re
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D

source_df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
print(source_df.head())
X = source_df.values

clusterNum = 3
k_means = KMeans(n_clusters = clusterNum)
k_means.fit(X)
labels = k_means.labels_
centers = k_means.cluster_centers_

print(labels)
print(centers)

source_df['Cluster'] = labels
X = source_df.values

print(X)

new_X = []
new_Y = []
new_Z = []
color = []

color_array = ['red', 'green', 'blue']

#for record in X:
#    new_X.append(record[0])
#    new_Y.append(record[1])
#    new_Z.append(record[4])
#    color.append(color_array[int(record[10])])

#plt.scatter(x = new_X, y = new_Y, c = color)

fig = plt.figure(None,(10.0, 10.0))
ax = fig.add_subplot(111, projection='3d')


for i in range(100):
    record = X[i]
    xs = record[0]
    ys = record[1] 
    zs = record[4] / 1000 / 100
    color = color_array[int(record[10])]
    ax.scatter(xs, ys, zs, c = color)

for record in centers:
    xs = record[0]
    ys = record[1]
    zs = record[4] / 1000 / 100
    color = 'black'
    ax.scatter(xs,ys,zs, c = color, s = 100)

ax.set_xlabel('Возраст')
ax.set_ylabel('Креатинфосфокиназа')
ax.set_zlabel('Тромбоциты * 100000')

plt.show()



filenames = ['chemistry_1.txt', 'chemistry_2.txt', 'chemistry_3.txt', 'chemistry_4.txt', 'chemistry_5.txt',
            'history_1.txt', 'history_2.txt', 'history_3.txt', 'history_4.txt', 'history_5.txt',
            'nature_1.txt', 'nature_2.txt', 'nature_3.txt', 'nature_4.txt', 'nature_5.txt']
            
l = ["" for i in range(15)]
for i in range(15):
    s = "data/" + filenames[i]
    with open(s, "r") as inf:
        l[i] = inf.read().lower() # читаем текст в нижнем регистре 
        l[i] = re.sub(r"[,\t;:\(\)\.\"«»“”\d\+\?\[\]\']*", r"", l[i]) #регулярками выпиливаем "левые символы"
        l[i] = re.sub(r"[\n]+", r" ", l[i]) #выпиливаем переносы строк, чтобы текст преобразовался в просто набор слов через пробел


d = {}
for i in range(15):
    text = word_tokenize(text=l[i])#получение списка всех слов
    result = []
    for word in text:
        if word in stopwords.words("english"): #удаляем шумовые слова английского языка в тексте
            continue
        if word not in result:
            result.append(word)
    d[i] = result

with open('test.txt', 'w') as file:
    for i in range(15):
            for word in d[i]:
                file.write(word + ' ') 
            file.write('\n------------------------------------------\n')


list = []

for item in d[14]:
    k = 0
    
    for i in range(14):
        if item in d[i]:
            k += 1
    if k > 5:
        list.append(item)     
print(list)

for i in range(15):
    for item in list:
        if item in d[i]:
            d[i].remove(item)
#print(d)
#выпиливаем слова которые очень часто встречаются в текстахх (5 и более раз)

def get_list_repeating_words(data, left, right):
    rep_words = []
    words = data[left]
    for word in words:
        count = 0
        for i in range(left,right):
            if word in data[i]:
                count += 1
        if count > 1:
            rep_words.append(word)
    return rep_words
        

list1 = get_list_repeating_words(d, 0, 4)

list2 = get_list_repeating_words(d, 5, 9)

list3 = get_list_repeating_words(d, 10, 14)

x1 = []
x2 = []
x3 = []

for i in range(15):
    k = 0
    for item in list1:# ищем кол-во слов в каждом из больших словарей текста
        if item in d[i]:
            k += 1
    x1.append(k)
    k = 0
    for item in list2:
        if item in d[i]:
            k += 1
    x2.append(k)
    k = 0
    for item in list3:
        if item in d[i]:
            k += 1
    x3.append(k)

table = pd.DataFrame({})
for i in range(len(x1)):
    table[i+1] = [x1[i], x2[i], x3[i]]
table.index = ["list1", "list2", "list3"]
table.head()


# dictionary of lists  
dict = {'List1': x1, 'List2': x2, 'List3': x3}  
    
df = pd.DataFrame(dict) 
  
X = df.values[:,0:]

clusterNum = 3
k_means = KMeans(n_clusters = clusterNum)
k_means.fit(X) 
labels = k_means.labels_
centers = k_means.cluster_centers_

df['Cluster'] = labels

figure = plt.figure(1, figsize = (8,6))
ax = Axes3D(figure, rect = [0, 0, 1, 1], elev = 20, azim = 120)

ax.set_xlabel("List1")
ax.set_ylabel("List2")
ax.set_zlabel("List3")
ax.scatter(X[:,0],X[:,1],X[:,2], c = labels.astype(np.float))

for center in centers:    
    ax.scatter(center[0], center[1] , center[2], c = 'black', s = 100)

plt.show()