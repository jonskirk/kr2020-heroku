#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# our kanji bitmap import utility
# we're expecting the bitmaps to be in a single CSV file called bitmaps.csv
# the first column contains the kanji, so we convert that to a one hot array with the same length as the kanjilist

# here's our set of kanji
# generated using rake get_bitmap_file_kanjilist
kanjilist = "⺀⺄⺈⺌⺍⺕⺝⺢⺤⺧⺮⺼⻊⻌⻏⻖⻗タノハマヨヰㄋ㇇㐮㑒㓁㔾㠯㦮䏍䒑一丁丂七丅丆三上丌且业丨丬中丰丶丷丸主乂乃乍乙乚九也亅予二于云五井亠亡交亥亦京人亻亼今付令余俞儿兀元兄兆兑入八公六共关其兼冂内冊冋再冏冓冖冫几凡凵凶出刀刂分刖力勹勺勿匂包匕化区十千卆卜占卩卬卯厶去又友反口古句召可台司各合同咅咼品員商啇喿囗土圣圭坴垂士壬壴夂复夕大天夫夭央奇女子孑宀寸寺専小少尗尞尸尺屮山川工巨己巴巾市干并幺广廴廾开弋弓彐彡彳心忄必戈戊戌我戸手扌支攵文斉斗斤方日旦旨早昔昜曲曽月有木未束東林果欠次止正歹殳毋母毎比毛氏氐水氵氷火灬烏爪爰父片牙牛牜犬犭玄玉王瓦甘生用甫田由申电畐疋疒癶白百皮皿監目直矛矢石示礻禸禺禾穴立竹米糸系缶罒羊羽老耂者而耳聿肀肉肖臣臤自至臼舌舛舟艮良色艹莫虍虫血行衣衤西覀見角言谷豆豕貝責赤走足身車辛辟辰辶酉釆里重金長門阜隹雨青非面革韋音頁風飛食飠首香馬骨高鬼魚鹿麦麻黄黒鼓鼻龶龷龸龹𠂇𠂉𠂤𠃊𠆢𠦝𠫓𡗗𦍌𦥑𧘇𩙿"

def get_batch(limit=None):
    # print "Reading in and transforming bitmap data..."
    df = pd.read_csv('/Users/jonskirk/Downloads/bitmaps.csv',header=None)
    # print(df.shape)

    data = df.values

    np.random.shuffle(data)

    # X is our pixel data - 32 x 32 = 1024, 0 or 1
    X = data[:, 1:]
    # X = data[:,1:785] / 255.0
    # print(X)

    # Y is our labels
    Y = convert_to_one_hot(data[:, 0])
    # print(Y)

    if limit is not None:
        X, Y = X[:limit], Y[:limit]

    return X, Y

# convert a list of kanji (ie, col 1 from our bitmaps file) to a one-hot matrix
def convert_to_one_hot(arr):

    # these are all three byte unicode chars
    # ret = np.zeros((arr.shape[0], len(strokes)/3))
    # no longer true with the addition of ˈ
    # wish python handled unicode as well as ruby!!
    ret = np.zeros((arr.shape[0], 403))

    for n in range(arr.shape[0]):
        # ret[n, kanjilist.index(arr[n])/3] = 1
        # ret[n, unicode(kanjilist,'utf8').find(unicode(arr[n],'utf8'))] = 1
        ret[n, kanjilist.find(arr[n])] = 1

    return ret

