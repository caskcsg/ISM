import json
import shutil

#Uniter
# with open('/workspace/project/UNITER/ann_t/train.jsonl','r',encoding='utf8')as fp:
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["identifier"] = str(dic["id"])
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         with open('/workspace/project/UNITER/ann_t/train_twitter.json','a',encoding='utf8')as fp:
#             #json.dump(dicc, fp)
#             json.dump(dicc, fp)
#             fp.write('\n')

# with open('/workspace/twitter/val.jsonl','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/twitter/ALBEF/val.json','a',encoding='utf8')as fp:
#         json.dump(a, fp)
# with open('/workspace/hateful_memes/dev_seen.jsonl','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/hateful_memes/ALBEF/dev_seen.json','a',encoding='utf8')as fp:
#         json.dump(a, fp)
# with open('/workspace/twitter/test.jsonl','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         dicc = {}
#         dicc["image"] = dic["id"]
#         dicc["sentence"] = dic["text"]
#         dicc["label"] = dic["label"]
#         a.append(dicc)
#     with open('/workspace/twitter/ALBEF/test.json','a',encoding='utf8')as fp:
#         json.dump(a, fp)

# with open('/workspace/multioff/ALBEF/train.json','r',encoding='utf8')as fp:
#     for line in fp.readlines():
#         dic = json.loads(line)
#         l = len(dic)
#         for i in range(l):
#             if dic[i]["label"] == 0:
#                 dicc = {}
#                 dicc["id"] = dic[i]["image"]
#                 dicc["img"] = "img/"+ dic[i]["image"]+".png"
#                 dicc["label"] = dic[i]["label"]
#                 dicc["text"] = dic[i]["sentence"]
#                 with open('/workspace/multioff/ALBEF/train0.json','a',encoding='utf8')as fp:
#                     json.dump(dicc, fp)
#                     fp.write('\n')
#             else :
#                 dicc = {}
#                 dicc["id"] = dic[i]["image"]
#                 dicc["img"] = "img/"+ dic[i]["image"]+".png"
#                 dicc["label"] = dic[i]["label"]
#                 dicc["text"] = dic[i]["sentence"]
#                 with open('/workspace/multioff/ALBEF/train1.json','a',encoding='utf8')as fp:
#                     json.dump(dicc, fp)
#                     fp.write('\n')


# with open('/workspace/multioff/ALBEF/test.json','r',encoding='utf8')as fp:
#     for line in fp.readlines():
#         dic = json.loads(line)
#         l = len(dic)
#         for i in range(l):
#             dicc = {}
#             dicc["id"] = dic[i]["image"]
#             dicc["img"] = "img/"+ dic[i]["image"]+".png"
#             dicc["label"] = dic[i]["label"]
#             dicc["text"] = dic[i]["sentence"]
#             with open('/workspace/multioff/ALBEF/test1.json','a',encoding='utf8')as fp:
#                 json.dump(dicc, fp)
#                 fp.write('\n')

# dicc={}
# with open('/workspace/twitter/ALBEF/train.json','r',encoding='utf8')as fp:
#     for line in fp.readlines():
#         dic = json.loads(line)
#         l = len(dic)
#         print(l)
#         for i in range(l):
#             li = dic[i]["sentence"].split(" ")
#             a = len(li)
#             if str(a) not in dicc.keys():
#                 dicc[str(a)] = 1
#             else :
#                 dicc[str(a)] += 1
# a2 = sorted(dicc.items(), key=lambda x: x[0])
# print(a2)

# import numpy as np

# file = np.load('/workspace/fast/output/allimages.npy', allow_pickle=True)
# print(file)
# np.savetxt('/workspace/fast/timestamps.txt',file)
            

# dicc={}
# with open('/workspace/multioff/ALBEF.json','r',encoding='utf8')as fp:
#     for line in fp.readlines():
#         dic = json.loads(line)
#         l = len(dic)
#         print(l)
#         for i in range(l):
#             li = dic[i]["image"]
            
# a2 = sorted(dicc.items(), key=lambda x: x[0])
# print(a2)
            


# import os 
# from os import path 
# #定义一个函数
# dic = {}
# def scaner_file (url, i):
#     #遍历当前路径下所有文件
#     file  = os.listdir(url)
#     for f in file:
#         #字符串拼接
#         fi = os.path.join(url, f)
#         a, _ = f.split(".png")
#         new = str(i)+".png"
#         if len(str(i)) < 3 :
#             if len(str(i)) == 1:
#                 new = '00' + str(i)+".png"
#             else :
#                 new = '0' + str(i)+".png"
#         n, _ = new.split(".png")         
#         dic[a] = n
#         i += 1
#         new_fi = os.path.join(url, new)
#         os.rename(fi,new_fi)
#     print(dic)
# #调用自定义函数
# i = 000
# scaner_file("/workspace/multioff/img_bak", i) 

# dic_c = {'uUc4OqX': '000', '2siNIoX': '001', 'RgdeucC': '002', 'IcRjmu4': '003', 'CdPJdPnVIAQqX9G': '004', 'GYmNXOy': '005', 'otKSntf': '006', 'WBYOXcj': '007', 
# '4ZKsrX3': '008', 'i0YUj7r': '009', 'E0JmmKR': '010', 'ZAloCy9': '011', '1JQk5NF': '012', '8WsB12c': '013', 'gaf6LVz': '014', '53U5Bca': '015', 'ERe2yR0': '016', 
# 'PAEZU3T': '017', 'qi0bYgG': '018', 'bnQlLeN': '019', '1Yj5Kns': '020', 'CUsbUT2': '021', '2d5AxL6': '022', 'qDnIIHA': '023', 'nWjooc9': '024', '4j4jraI': '025', 
# 'hDJHnuW': '026', 'ubeo3Ya': '027', '5spRvtw': '028', 'UK1yqyr': '029', 'xlV0M7Q': '030', 'CsdehGlWgAAv2HI': '031', 'CZrwNVe': '032', 'bxT32op': '033', 
# 'JZOT6IU': '034', '7CIR9VI': '035', 'X7mEejq': '036', 'DB1R0kO': '037', 'cWdiU5k': '038', 'dOjn3Mo': '039', 'ddkILbG': '040', 'IVgSveX': '041', 'DAJu7lB': '042', 
# 'Pv3M76n': '043', '34nLFqO': '044', '2iaC1qq': '045', 'Eq9f1cu': '046', 'RhODpdf': '047', 'p3u6fbf': '048', 'mycUTal': '049', 'LH0waGz': '050', 'mpA6ezG': '051', 
# 'CWxcn8cWcAEOWqO': '052', 'CqMB3fnW8AAsYQp': '053', 'TpIZoZr': '054', 'EuTcoU6': '055', '8mhcYsP': '056', 'te0QdLn': '057', 'cljHNxO': '058', 'CdXi98yXEAA33CR': '059', 
# 'KbTk7Rq': '060', '7rNwjnv': '061', 'gtxLH3X': '062', 'k7472hu': '063', '8mROrGY': '064', 'zFnJ1x3': '065', 'BuTZiB-IAAIO3am': '066', 'uelxFPQ': '067', 'NLdr9yb': '068', 
# 'oyjvlEa': '069', 'p3MNKdN': '070', 'iZR0Ych': '071', '7Q0ReMA': '072', '0N5ZKcH': '073', 'YIi9Bga': '074', 'Qj9x2wz': '075', '3wirGHJ': '076', 'SGQOyjW': '077', 
# 'odYglAk': '078', '9FjUVia': '079', 'WSy7NGq': '080', 'gSYEayq': '081', '80NRcEf': '082', 'D1uY1Ye': '083', 'k5DLjAd': '084', 'wkkA7rU': '085', 'FYijK60': '086', 
# 'YJ1QbDX': '087', '0nF9BpI': '088', 'ogrQSpx': '089', '2myYC1v': '090', 'rOvVkF1': '091', 'uTJaugb': '092', 'WQjyHN1': '093', '5WXHaW4': '094', 'h45acTC': '095', 
# 'CcsU2lQUEAEoeuW': '096', 'NypZ28n': '097', '1nSSB9S': '098', 'pOkTegk': '099', 'MOMyvfC': '100', 'gy3AJXJ': '101', 'NwQNeBd': '102', 'KCHMiA2': '103', 
# 'bGZd5db': '104', 'qMmIGUr': '105', 'yZLLqn4': '106', 'GbhHJH6': '107', 'VSbm5Ul': '108', '00DjNzR': '109', 'UTDvvsp': '110', 'puHulqv': '111', 'qq8wzMR': '112', 
# '76vgP7x': '113', 'EZMV9D9': '114', 'risXjE4': '115', 'PCnauaB': '116', 'CguD5hk': '117', 'T9pQAFP': '118', 'Xx9KwVV': '119', 'z5sSuk3': '120', 'Er3pBdJ': '121', 
# 'uTYLSi7': '122', 'nNYop46': '123', 'uGV2kpi': '124', 'lJVnWF2': '125', 'QEM6dG2': '126', 'NtpMxK9': '127', 'phyDxAb': '128', 'XwccSUy': '129', 'iOyEYMl': '130', 
# 'CboHZesW8AACcdP': '131', 'LEMUxIr': '132', 'fU7SAmZ': '133', 'qHS51uj': '134', 'HnYtR6j': '135', 'naXRSpK': '136', 'qfYlr30': '137', 'CjRhYGU': '138', 'Ar6lu6P': '139', 
# '5hN4XOF': '140', 'JRpm6xn': '141', 'CFa0OLu': '142', 'D9Hn7ll': '143', 'IkosYmM': '144', '1YQOnyG': '145', 'xOFyAk8': '146', 'aaQaOEz': '147', '88B7ag2': '148', 
# 'CPVA4poWoAAhOs6': '149', '12XLnzK': '150', 'ykGRBWn': '151', 'pnJqmzw': '152', 'v2dOGRp': '153', 'GaL4tca': '154', 'O1JHlod': '155', 'XtxfPFR': '156', 'xLXvr9t': '157', 
# 'TLQF1bX': '158', 'k6h3IIK': '159', 'rklsz23': '160', 'yLF1GSc': '161', 'm4oY5vK': '162', 'gsH7Gel': '163', 'zQPb3nD': '164', 'bHyYl3j': '165', 'fqSHGCD': '166', 
# 'j9AOSo2': '167', 'kzLhUFd': '168', 'CjZPiloXAAA1v_W': '169', 'WvnX54P': '170', 'eP24hnr': '171', '7lqO9Qr': '172', 'wZk5FIY': '173', 'FsAbNtn': '174', 'fkpqfLB': '175', 
# 'NpsCjMn': '176', 'WPnY6NN': '177', '9CYSG3P': '178', 'V2wUrRj': '179', 'vbrzXLV': '180', 'kzNSl6e': '181', 'MfHuFve': '182', 'yHg7DkU': '183', 'vTulmoz': '184', 
# 'mZmgkws': '185', 'DZ2uQ50': '186', 'pW6hm0W': '187', '7juAkcv': '188', 'Svfe8vm': '189', 'kxA5NIk': '190', 'zq7bXGu': '191', '9gSRNmP': '192', 'A0wcV4g': '193', 
# 'sOYYbyS': '194', '18nGcfg': '195', 'AY91zJl': '196', 'jyxHhiB': '197', 'cnUn6I4': '198', 'CbYe-4SW8AAmvbJ': '199', '0Jzts4J': '200', 'Nkvcr8I': '201', 'AJvEoxw': 
# '202', 'eeuDu6t': '203', 'Rb48yZM': '204', 'uDZH05A': '205', 's3iOROE': '206', 'kUlXoaa': '207', 'giep513': '208', 'vzQSRVt': '209', 'DBSMGfC': '210', 'rgYqE0y': '211', 
# 'vHFo7dO': '212', '9diI55h': '213', 'VMtdwll': '214', 'ZwFfyVJ': '215', 'W3jhTxL': '216', 'QbYt8jJ': '217', 'Wwe2NdZ': '218', 'pgVlfQb': '219', 'N26emlc': '220', 
# '9mITvJA': '221', 'p95qrbn': '222', 'zKbUmLB': '223', 'pyCWWr7': '224', 'JU9LSXi': '225', 'JxTc0EC': '226', 'SiKwS84': '227', 'EIehYa1': '228', 'l6SOQGq': '229', 
# 'DmcdsLa': '230', '5YMQ6wi': '231', 'O3KNvgz': '232', 'OhXur08': '233', 'TbOyNzW': '234', 'oEGSd01': '235', '4jYyIBw': '236', '43LJZG9': '237', 'WwN5X3R': '238', 
# '94anjQG': '239', 'KmlWyX7': '240', 'ZOSUjXw': '241', 'Mmtyp8J': '242', 'NpDM9E8': '243', 'aTX7lyO': '244', 'omVP1Lw': '245', '873SgEx': '246', 'hZ9k5JE': '247', 
# 'xtvEjIx': '248', 'JsZdMIq': '249', 'mHAg22Y': '250', 'nnEBoVw': '251', '6eRZSgn': '252', 'iMCzClJ': '253', 'yDO396v': '254', 'Wk0Wz2D': '255', 'mn6RF0J': '256', 
# 'CagRUffUcAExvLF': '257', 'xrh5RQD': '258', 'CpGttnLXYAATNHh': '259', 'UzN2sz1': '260', 'SazlDcs': '261', 'MLw5fUA': '262', 'f7y8o6x': '263', 'taecqiR': '264', 
# 'NdIEduW': '265', 'RfpjmXp': '266', 's8bBJYS': '267', 'Z1VeVXi': '268', 'Qlm1siZ': '269', 'mErGXnG': '270', 'W0Nh6hc': '271', 'CVf4HDmWcAE_Q4m': '272', 
# 'sG2GSpb': '273', '95yFB4s': '274', 'CcmXJPoUcAAkauM': '275', 'bcX0n1K': '276', 'TL8WSKV': '277', '9xksTiP': '278', 'BScSZDe': '279', 'u4QjzUI': '280', 
# 'gdHUw8d': '281', 'MQqsZvW': '282', 'THX9f3n': '283', 'SPKD3JU': '284', 'BkZ356ICIAAEHId': '285', 'fXXcSD2': '286', 'YGkNITr': '287', 'iMMNq': '288', 
# 'jVHe747': '289', '8ZOPW62': '290', 'Oa1JNby': '291', 'CuXk-UQWYAAs43F': '292', 'IhjXkUr': '293', '8oKcC3I': '294', 'dPkUsp1': '295', 'b3N3bmV': '296', 
# 'Xxc4mjq': '297', '0JM56ut': '298', 'V9SBG1K': '299', 'CfU-_jqW4AAmTBw': '300', 'jQsh9vC': '301', '27S1feu': '302', 'yJZO4mQ': '303', 'X0RNiHr': '304', 
# 'QTNlTQo': '305', '27nZ2PK': '306', 'B0rKblC': '307', 'A5t5Zkf': '308', 'VlcI2AV': '309', 'qKM2ucu': '310', 'gSIuTfF': '311', 'Ktvna1u': '312', 
# 'CW7tNBQU0AIADyq': '313', 'mvfGs4u': '314', 'JvY4wYr': '315', 't4xWpee': '316', 'M6wyDw8': '317', 'z1gozo1': '318', 'rGPad5Y': '319', '6r16i0p': '320', 'bCuLRM2': '321', 'oHjgmJS': '322', 'TFRcH6Q': '323', 'eJqdhnY': '324', 'swzhNs4': '325', '6L5ohmp': '326', 'QonIrvr': '327', 'ybxXd8R': '328', 'CVjO3Sz': '329', 'rOr3fxM': '330', 'tKPYO8r': '331', 'D1xG5': '332', 'Pn18AKw': '333', 'SWxsby2': '334', '1iLg1U1': '335', 'I84r9TR': '336', 'KZoud4R': '337', 'KnLJwdq': '338', 'D4k0puo': '339', 'nm9n7k0': '340', 'CMQqhImUkAAKeno': '341', '3YtnZiU': '342', '5W8eGmw': '343', 'krkcbjy': '344', 's972ZwU': '345', '8UJlZ3J': '346', 'lv3JsRq': '347', 'Hu8KaNu': '348', 'NwfrHyS': '349', 'FrLLfI8': '350', 'p2ciRYD': '351', 'R2jWn36': '352', '9h78t5i': '353', 'gZWRvf8': '354', '2MXmVri': '355', 'FuuH1LD': '356', 'okIQiIu': '357', 'k6x2h2d': '358', '474EyYj': '359', 'bmM6pSU': '360', 'K8v985Z': '361', 'MhnioLP': '362', 'EhpW75N': '363', 'se7MJYj': '364', 'CuvsjyLWgAAXnz9': '365', 'tINjUCc': '366', 'mOHGolk': '367', 'EsepD5Q': '368', 'CaQrRikUYAA3VrV': '369', 'J7KwOfQ': '370', 'jn4oI6f': '371', '6G0PNe1': '372', 'KMa8L39': '373', '3CbPlsi': '374', 'Y85uWnm': '375', 'T7sOpjA': '376', 'x9dwWj9': '377', 'p8cqujC': '378', '7P6yB0D': '379', 'RSTEno4': '380', 'CkhmsCi': '381', 'nlzn1IN': '382', 'mVDgP9C': '383', '8oOhdGX': '384', 'VwfCCk1': '385', '2T7fEop': '386', 'Crp2aRz': '387', 'MBdsW19': '388', 'kixSnlQ': '389', 'oQWC8Od': '390', 'ezoih5p': '391', 'dcjraqN': '392', '2UT3Ips': '393', 'COkaPpN': '394', 'mrjFNGX': '395', '9dArUsU': '396', 'NsCDfKJ': '397', 'oRz3Ms8': '398', '0WBXRH9': '399', '55A2AUa': '400', 'a1F6RKt': '401', 'GaQNDn2': '402', 'bspb4sM': '403', 'ET5rdX2': '404', 'OH0Nuxb': '405', 'QHHMKB3': '406', 'QebeqSh': '407', '9A372Gl': '408', '95L0GB6': '409', 'yiS7xVy': '410', 'iutC6Qy': '411', 'aL1hjkE': '412', 'qv6y9xq': '413', '8NMWu57': '414', 'NssAP3V': '415', 'x76JHsa': '416', '1f2zSW8': '417', 'qERLZTk': '418', 'hV8hDZV': '419', 'ZbqRtuL': '420', 'nHevBnI': '421', 'T4u74mL': '422', 'AQFwrJx': '423', 'ZWajHQq': '424', 'CaqKlOXWIAAvy8L': '425', '91sHVxQ': '426', '7S6NEVD': '427', 'meTteYg': '428', 'PdPdKfd': '429', 'VNZBamf': '430', 'QLQ4taO': '431', 'HSVyqdL': '432', 'lVOjKnR': '433', 'gOHWdbe': '434', 'MmdEDKF': '435', 'i1KC0aU': '436', 'h6Pkqkr': '437', 'TOuTChz': '438', 'mVjzFQY': '439', 'WiqI5X3': '440', 'iZgFwkB': '441', 'h2nKIqJ': '442', '88pU7QL': '443', '0EwB4LT': '444', '9BmlvFy': '445', '0vqvDs5': '446', '3Zz0vux': '447', 'BWopGu8': '448', 'HUU1Fay': '449', 'CIFrUHV': '450', '6faN492': '451', 'TgSYoWe': '452', 'b3IhU9Z': '453', 'VvMdZpV': '454', 'VoNgxAD': '455', '0P5i3yI': '456', 'CfUTJmj': '457', '2Qo03El': '458', 'Lo2ylA3': '459', 'zAUFD7y': '460', 'XTg6oYN': '461', 'AGGHufY': '462', 'D9Zr2dU': '463', 'QnLfqfX': '464', 'sKiuQVf': '465', 'T2Jt2ry': '466', 'D40IF3q': '467', 'CdsdCaHXIAEw1An': '468', 'akXZYL5': '469', 'Q29wRyg': '470', 'm85k7GR': '471', '7wKCNn2': '472', 'kO0gHF2': '473', '05nBEvh': '474', 'lR6IYpG': '475', 'PpxbGZC': '476', 'Vi4Kjyx': '477', 'TjsILvp': '478', 'KiYSMvi': '479', 'fW7CAvO': '480', 'LE4FyLF': '481', 'mSA9fTA': '482', 'ayNSttp': '483', 'SZTu5uv': '484', 'CySpKqI': '485', 'CibcteKWsAEgqh0': '486', '5vVB395': '487', 'st6NozD': '488', 'AthUmaI': '489', 'qSyop93': '490', 'XoR7xtd': '491', 'X2dmDrp': '492', 'VYZEg9A': '493', 'ykaOeQZ': '494', 'xJiXDnY': '495', 'XYERAZg': '496', 'N4EIWJX': '497', '0SvkQMd': '498', '8Fwa0DR': '499', 'jkdwXzP': '500', 'sG8faWv': '501', 'FvlpogZ': '502', '7eTNqFK': '503', '0bOKK62': '504', 'mYU8r9Y': '505', 'qnRUXr8': '506', 'JYTdhl9': '507', 'hOxu3fK': '508', '6jA3QAb': '509', 'yGHKtzg': '510', 'zl617iP': '511', 'WcXzDk2': '512', 'zoIImWJ': '513', 'u9FeEuE': '514', 'hoApJnq': '515', 'L6WFxJs': '516', 'qoZxIKO': '517', 'LJ3r8Gy': '518', 'C0JFBbU': '519', '9kZb6UD': '520', 'DDWmGxP': '521', 'BmKzn6gIYAIwC-U': '522', 'dOr1Ljj': '523', 'l7UFL6x': '524', 'oqt51kd': '525', 'iAztwx4': '526', '0p7HM8n': '527', '8jiJ4Zt': '528', '5hhUFqA': '529', 'C5mm2gl': '530', 'RX8t7Jd': '531', 'y898T2X': '532', 'ZffTHk2': '533', '4LoGRB5': '534', '5aomAQA': '535', 'Cc8PZCkXIAAmG0f': '536', 'oKD3FG3': '537', 'Hf65xSJ': '538', 'KIiW0Lw': '539', 'ugTPNMd': '540', 'dZVtSfm': '541', 'jAi3iI1': '542', 'V6MP5lX': '543', '4KXGjz0': '544', 'Cm96pUGVMAA7HQz': '545', '9HHSnfe': '546', 'Okb6dJB': '547', 'zr5VeQM': '548', 's9Z2Han': '549', '9n1Tjiz': '550', 'NDFCXLG': '551', 'l0P4MzC': '552', 'ArmX82G': '553', 'pEPz8iF': '554', 'sbWUiil': '555', 'wK9e9jg': '556', 'zOImoh5': '557', 'BCah7Wz': '558', 'RbOxrFZ': '559', 'uDbb5Gk': '560', 'MZQc8jz': '561', '4j07SmD': '562', 'Nbo6iZU': '563', '8Djb33H': '564', 'nEE5sjy': '565', 'Zh4CZSe': '566', 'eI2N5iQ': '567', 'WDUgdRk': '568', 'bDIJ7sx': '569', 'hF5Bi0f': '570', 'JbWv3sM': '571', 'XniGGU5': '572', 'CwHWOROW8AA09nd': '573', 'zZIOwqg': '574', 'VOiuzA6': '575', '0Te3Qys': '576', 'L7jzD7i': '577', 'fsgozVS': '578', 'W5DXy2F': '579', 'BVzw01E': '580', 'QLgpXMT': '581', '0UQh5Eo': '582', 'eMSADPX': '583', 'nefjbtQ': '584', '4q1tono': '585', '2aiRLNR': '586', 'fyAh3I0': '587', '2NS27kx': '588', 'iQ0COdD': '589', 'Cco_BiaUMAAla6m': '590', 'mZvaiMI': '591', 'xYTrJJp': '592', 'sW3bJWW': '593', 'T35mICA': '594', 'wKSn3TS': '595', '6WELqLH': '596', 'CvOg2va': '597', 'iyTe9SI': '598', 'wgVTz9g': '599', '3fsUJvp': '600', 'WSdFOAD': '601', 'JBUKM8i': '602', 'Ce0OU-iUIAALkmn': '603', 'WyNvTye': '604', 'gbt5ETH': '605', 'bpc3wap': '606', '5oT4c6d': '607', 'XMReWCt': '608', 'Ob1Q9v8': '609', 'D65y7fq': '610', 'ZEjNdo0': '611', 'DLr1Uo7': '612', 'jQ6NQqx': '613', 'LAPqPzX': '614', 'ZMaAKc9': '615', 'aXJH7EU': '616', 'gK5tKHN': '617', '2tuVUzz': '618', 'QNB8H2s': '619', 'SXvE9sL': '620', 'zRgnrbT': '621', 'uluP5WP': '622', '6V4OrKm': '623', 'nsEOOZO': '624', 'bUfWNUp': '625', 'zPT9dFf': '626', 'B_LgxAvVIAA_lu8': '627', 'Cf87U8rW4AAtlvE': '628', 'ewKXYvk': '629', 'AMzup5d': '630', 'ztdf8dZ': '631', 'HyQBPvH': '632', 'TyYDiSx': '633', 'BXs0Wy8': '634', 'QmLaggk': '635', 'fmTCPZO': '636', 'BVzBIgy': '637', '1DY6I9Q': '638', 'ExlQpoj': '639', 'vsFSedb': '640', 'aKBYqSs': '641', 'xGBX7zN': '642', '8ItzKTN': '643', 'Bq3Oe2TIAAAUgik': '644', 'Wl32YJG': '645', '2TAk7vi': '646', 'LW5TKwe': '647', 'f0kDhU7': '648', 'WVeMKB9': '649', 'zvdzU3y': '650', '3cvbmM0': '651', 'OA7aCgM': '652', 'WZ1NLSt': '653', 'JT26Var': '654', 'Im9iERE': '655', 'gzWfYg3': '656', 'yBjFq4v': '657', 'wzOK4S3': '658', 'YW7mx62': '659', 'm0FKJHM': '660', 'V7Ie6Os': '661', 'so4V8X7': '662', 'otfhxeo': '663', 'd5ZqKJ5': '664', '6uTXEgh': '665', 'qZ8nvyF': '666', 'H6vd1DD': '667', 'cnkpnzp': '668', '4hxCAfW': '669', 'tMzNcad': '670', 'H0Bh7VP': '671', 'Rqzw9Co': '672', 'Nyo6x2Y': '673', 'Tf1mhKt': '674', 'pdaFFC4': '675', '5kSMRv3': '676', 'we4hhWi': '677', 'X6BtuWV': '678', 'C2Rjlth': '679', 'VOFh4qX': '680', '7COAAL3': '681', 'NyP2PbK': '682', 'whhrfSE': '683', 'qHkGmLt': '684', 'fdZpYpb': '685', 'ruSN6zI': '686', 'xyi7yMK': '687', '3bqSgzm': '688', 'QQnZqPB': '689', 'hLMgzsp': '690', 'H7Ob55y': '691', 'rDy2M4O': '692', 'IpwuEVr': '693', 'orV6OVk': '694', '5Oznxbq': '695', '1BeUIMs': '696', 'DheGAH3': '697', 'dCBEORM': '698', 'rmlkOMu': '699', 'DNwiCOI': '700', 'CcFmlzLW8AAxHkM': '701', 'eRUFvRF': '702', 'xQkfJse': '703', '5NWPOQp': '704', 'jG51M2l': '705', 'XX28NwO': '706', 'fLxsrjJ': '707', 'IWXLlFJ': '708', 'nqfekut': '709', 'JHF6hFI': '710', 's1qivuu': '711', 'wwEGRrn': '712', 'TqFVPVE': '713', 'k129f29': '714', 'zs6maXI': '715', 'LkX8Ei3': '716', '6NFcW7v': '717', 'PjZa9M1': '718', 'lPguwLG': '719', '16vTnwg': '720', 'BNPocHU': '721', '209BoJK': '722', '2pt7Wkd': '723', 'JYTlRZK': '724', 'Lsjt4TJ': '725', 'Ci45dogWEAA2bSb': '726', 'TWjXbPK': '727', 'ysAk127': '728', 'cAm7azy': '729', 'pvCTAHi': '730', 'ky2vvAw': '731', 'tcXzxyV': '732', 'fC5euc2': '733', 'koxcr8J': '734', 'JK1w4yI': '735', 'FSeqQG9': '736', 'h92g9FL': '737', 'yWd3NEB': '738', 'XqB2jSY': '739', 'EMshcj9': '740', 'aK7Mzo4': '741', 'LeHuK4y': '742', 'CkbzAqAUoAUJ311': '743', '7Lg5Rd2': '744', 'VAJZJJV': '745', 'CecBrceWQAQR19j': '746', 'rZGXxM4': '747', 'WDSz8Sa': '748'}


# with open('/workspace/multioff/ALBEF/test.json','r',encoding='utf8')as fp:
#     for line in fp.readlines():
#         dic = json.loads(line)
#         l = len(dic)
#         for i in range(l):
#             dicc = {}
#             dicc["id"] = dic_c[dic[i]["image"]]
#             dicc["img"] = "img/"+ dic_c[dic[i]["image"]]+".png"
#             dicc["label"] = dic[i]["label"]
#             dicc["text"] = dic[i]["sentence"]
#             with open('/workspace/multioff/ALBEF/test_b.json','a',encoding='utf8')as fp:
#                 json.dump(dicc, fp)
#                 fp.write('\n')


# dic = {'uUc4OqX': '000', '2siNIoX': '001', 'RgdeucC': '002', 'IcRjmu4': '003', 'CdPJdPnVIAQqX9G': '004', 'GYmNXOy': '005', 'otKSntf': '006', 'WBYOXcj': '007', 
# '4ZKsrX3': '008', 'i0YUj7r': '009', 'E0JmmKR': '010', 'ZAloCy9': '011', '1JQk5NF': '012', '8WsB12c': '013', 'gaf6LVz': '014', '53U5Bca': '015', 'ERe2yR0': '016', 
# 'PAEZU3T': '017', 'qi0bYgG': '018', 'bnQlLeN': '019', '1Yj5Kns': '020', 'CUsbUT2': '021', '2d5AxL6': '022', 'qDnIIHA': '023', 'nWjooc9': '024', '4j4jraI': '025', 
# 'hDJHnuW': '026', 'ubeo3Ya': '027', '5spRvtw': '028', 'UK1yqyr': '029', 'xlV0M7Q': '030', 'CsdehGlWgAAv2HI': '031', 'CZrwNVe': '032', 'bxT32op': '033', 
# 'JZOT6IU': '034', '7CIR9VI': '035', 'X7mEejq': '036', 'DB1R0kO': '037', 'cWdiU5k': '038', 'dOjn3Mo': '039', 'ddkILbG': '040', 'IVgSveX': '041', 'DAJu7lB': '042', 
# 'Pv3M76n': '043', '34nLFqO': '044', '2iaC1qq': '045', 'Eq9f1cu': '046', 'RhODpdf': '047', 'p3u6fbf': '048', 'mycUTal': '049', 'LH0waGz': '050', 'mpA6ezG': '051', 
# 'CWxcn8cWcAEOWqO': '052', 'CqMB3fnW8AAsYQp': '053', 'TpIZoZr': '054', 'EuTcoU6': '055', '8mhcYsP': '056', 'te0QdLn': '057', 'cljHNxO': '058', 'CdXi98yXEAA33CR': '059', 
# 'KbTk7Rq': '060', '7rNwjnv': '061', 'gtxLH3X': '062', 'k7472hu': '063', '8mROrGY': '064', 'zFnJ1x3': '065', 'BuTZiB-IAAIO3am': '066', 'uelxFPQ': '067', 'NLdr9yb': '068', 
# 'oyjvlEa': '069', 'p3MNKdN': '070', 'iZR0Ych': '071', '7Q0ReMA': '072', '0N5ZKcH': '073', 'YIi9Bga': '074', 'Qj9x2wz': '075', '3wirGHJ': '076', 'SGQOyjW': '077', 
# 'odYglAk': '078', '9FjUVia': '079', 'WSy7NGq': '080', 'gSYEayq': '081', '80NRcEf': '082', 'D1uY1Ye': '083', 'k5DLjAd': '084', 'wkkA7rU': '085', 'FYijK60': '086', 
# 'YJ1QbDX': '087', '0nF9BpI': '088', 'ogrQSpx': '089', '2myYC1v': '090', 'rOvVkF1': '091', 'uTJaugb': '092', 'WQjyHN1': '093', '5WXHaW4': '094', 'h45acTC': '095', 
# 'CcsU2lQUEAEoeuW': '096', 'NypZ28n': '097', '1nSSB9S': '098', 'pOkTegk': '099', 'MOMyvfC': '100', 'gy3AJXJ': '101', 'NwQNeBd': '102', 'KCHMiA2': '103', 
# 'bGZd5db': '104', 'qMmIGUr': '105', 'yZLLqn4': '106', 'GbhHJH6': '107', 'VSbm5Ul': '108', '00DjNzR': '109', 'UTDvvsp': '110', 'puHulqv': '111', 'qq8wzMR': '112', 
# '76vgP7x': '113', 'EZMV9D9': '114', 'risXjE4': '115', 'PCnauaB': '116', 'CguD5hk': '117', 'T9pQAFP': '118', 'Xx9KwVV': '119', 'z5sSuk3': '120', 'Er3pBdJ': '121', 
# 'uTYLSi7': '122', 'nNYop46': '123', 'uGV2kpi': '124', 'lJVnWF2': '125', 'QEM6dG2': '126', 'NtpMxK9': '127', 'phyDxAb': '128', 'XwccSUy': '129', 'iOyEYMl': '130', 
# 'CboHZesW8AACcdP': '131', 'LEMUxIr': '132', 'fU7SAmZ': '133', 'qHS51uj': '134', 'HnYtR6j': '135', 'naXRSpK': '136', 'qfYlr30': '137', 'CjRhYGU': '138', 'Ar6lu6P': '139', 
# '5hN4XOF': '140', 'JRpm6xn': '141', 'CFa0OLu': '142', 'D9Hn7ll': '143', 'IkosYmM': '144', '1YQOnyG': '145', 'xOFyAk8': '146', 'aaQaOEz': '147', '88B7ag2': '148', 
# 'CPVA4poWoAAhOs6': '149', '12XLnzK': '150', 'ykGRBWn': '151', 'pnJqmzw': '152', 'v2dOGRp': '153', 'GaL4tca': '154', 'O1JHlod': '155', 'XtxfPFR': '156', 'xLXvr9t': '157', 
# 'TLQF1bX': '158', 'k6h3IIK': '159', 'rklsz23': '160', 'yLF1GSc': '161', 'm4oY5vK': '162', 'gsH7Gel': '163', 'zQPb3nD': '164', 'bHyYl3j': '165', 'fqSHGCD': '166', 
# 'j9AOSo2': '167', 'kzLhUFd': '168', 'CjZPiloXAAA1v_W': '169', 'WvnX54P': '170', 'eP24hnr': '171', '7lqO9Qr': '172', 'wZk5FIY': '173', 'FsAbNtn': '174', 'fkpqfLB': '175', 
# 'NpsCjMn': '176', 'WPnY6NN': '177', '9CYSG3P': '178', 'V2wUrRj': '179', 'vbrzXLV': '180', 'kzNSl6e': '181', 'MfHuFve': '182', 'yHg7DkU': '183', 'vTulmoz': '184', 
# 'mZmgkws': '185', 'DZ2uQ50': '186', 'pW6hm0W': '187', '7juAkcv': '188', 'Svfe8vm': '189', 'kxA5NIk': '190', 'zq7bXGu': '191', '9gSRNmP': '192', 'A0wcV4g': '193', 
# 'sOYYbyS': '194', '18nGcfg': '195', 'AY91zJl': '196', 'jyxHhiB': '197', 'cnUn6I4': '198', 'CbYe-4SW8AAmvbJ': '199', '0Jzts4J': '200', 'Nkvcr8I': '201', 'AJvEoxw': 
# '202', 'eeuDu6t': '203', 'Rb48yZM': '204', 'uDZH05A': '205', 's3iOROE': '206', 'kUlXoaa': '207', 'giep513': '208', 'vzQSRVt': '209', 'DBSMGfC': '210', 'rgYqE0y': '211', 
# 'vHFo7dO': '212', '9diI55h': '213', 'VMtdwll': '214', 'ZwFfyVJ': '215', 'W3jhTxL': '216', 'QbYt8jJ': '217', 'Wwe2NdZ': '218', 'pgVlfQb': '219', 'N26emlc': '220', 
# '9mITvJA': '221', 'p95qrbn': '222', 'zKbUmLB': '223', 'pyCWWr7': '224', 'JU9LSXi': '225', 'JxTc0EC': '226', 'SiKwS84': '227', 'EIehYa1': '228', 'l6SOQGq': '229', 
# 'DmcdsLa': '230', '5YMQ6wi': '231', 'O3KNvgz': '232', 'OhXur08': '233', 'TbOyNzW': '234', 'oEGSd01': '235', '4jYyIBw': '236', '43LJZG9': '237', 'WwN5X3R': '238', 
# '94anjQG': '239', 'KmlWyX7': '240', 'ZOSUjXw': '241', 'Mmtyp8J': '242', 'NpDM9E8': '243', 'aTX7lyO': '244', 'omVP1Lw': '245', '873SgEx': '246', 'hZ9k5JE': '247', 
# 'xtvEjIx': '248', 'JsZdMIq': '249', 'mHAg22Y': '250', 'nnEBoVw': '251', '6eRZSgn': '252', 'iMCzClJ': '253', 'yDO396v': '254', 'Wk0Wz2D': '255', 'mn6RF0J': '256', 
# 'CagRUffUcAExvLF': '257', 'xrh5RQD': '258', 'CpGttnLXYAATNHh': '259', 'UzN2sz1': '260', 'SazlDcs': '261', 'MLw5fUA': '262', 'f7y8o6x': '263', 'taecqiR': '264', 
# 'NdIEduW': '265', 'RfpjmXp': '266', 's8bBJYS': '267', 'Z1VeVXi': '268', 'Qlm1siZ': '269', 'mErGXnG': '270', 'W0Nh6hc': '271', 'CVf4HDmWcAE_Q4m': '272', 
# 'sG2GSpb': '273', '95yFB4s': '274', 'CcmXJPoUcAAkauM': '275', 'bcX0n1K': '276', 'TL8WSKV': '277', '9xksTiP': '278', 'BScSZDe': '279', 'u4QjzUI': '280', 
# 'gdHUw8d': '281', 'MQqsZvW': '282', 'THX9f3n': '283', 'SPKD3JU': '284', 'BkZ356ICIAAEHId': '285', 'fXXcSD2': '286', 'YGkNITr': '287', 'iMMNq': '288', 
# 'jVHe747': '289', '8ZOPW62': '290', 'Oa1JNby': '291', 'CuXk-UQWYAAs43F': '292', 'IhjXkUr': '293', '8oKcC3I': '294', 'dPkUsp1': '295', 'b3N3bmV': '296', 
# 'Xxc4mjq': '297', '0JM56ut': '298', 'V9SBG1K': '299', 'CfU-_jqW4AAmTBw': '300', 'jQsh9vC': '301', '27S1feu': '302', 'yJZO4mQ': '303', 'X0RNiHr': '304', 
# 'QTNlTQo': '305', '27nZ2PK': '306', 'B0rKblC': '307', 'A5t5Zkf': '308', 'VlcI2AV': '309', 'qKM2ucu': '310', 'gSIuTfF': '311', 'Ktvna1u': '312', 
# 'CW7tNBQU0AIADyq': '313', 'mvfGs4u': '314', 'JvY4wYr': '315', 't4xWpee': '316', 'M6wyDw8': '317', 'z1gozo1': '318', 'rGPad5Y': '319', '6r16i0p': '320', 'bCuLRM2': '321', 'oHjgmJS': '322', 'TFRcH6Q': '323', 'eJqdhnY': '324', 'swzhNs4': '325', '6L5ohmp': '326', 'QonIrvr': '327', 'ybxXd8R': '328', 'CVjO3Sz': '329', 'rOr3fxM': '330', 'tKPYO8r': '331', 'D1xG5': '332', 'Pn18AKw': '333', 'SWxsby2': '334', '1iLg1U1': '335', 'I84r9TR': '336', 'KZoud4R': '337', 'KnLJwdq': '338', 'D4k0puo': '339', 'nm9n7k0': '340', 'CMQqhImUkAAKeno': '341', '3YtnZiU': '342', '5W8eGmw': '343', 'krkcbjy': '344', 's972ZwU': '345', '8UJlZ3J': '346', 'lv3JsRq': '347', 'Hu8KaNu': '348', 'NwfrHyS': '349', 'FrLLfI8': '350', 'p2ciRYD': '351', 'R2jWn36': '352', '9h78t5i': '353', 'gZWRvf8': '354', '2MXmVri': '355', 'FuuH1LD': '356', 'okIQiIu': '357', 'k6x2h2d': '358', '474EyYj': '359', 'bmM6pSU': '360', 'K8v985Z': '361', 'MhnioLP': '362', 'EhpW75N': '363', 'se7MJYj': '364', 'CuvsjyLWgAAXnz9': '365', 'tINjUCc': '366', 'mOHGolk': '367', 'EsepD5Q': '368', 'CaQrRikUYAA3VrV': '369', 'J7KwOfQ': '370', 'jn4oI6f': '371', '6G0PNe1': '372', 'KMa8L39': '373', '3CbPlsi': '374', 'Y85uWnm': '375', 'T7sOpjA': '376', 'x9dwWj9': '377', 'p8cqujC': '378', '7P6yB0D': '379', 'RSTEno4': '380', 'CkhmsCi': '381', 'nlzn1IN': '382', 'mVDgP9C': '383', '8oOhdGX': '384', 'VwfCCk1': '385', '2T7fEop': '386', 'Crp2aRz': '387', 'MBdsW19': '388', 'kixSnlQ': '389', 'oQWC8Od': '390', 'ezoih5p': '391', 'dcjraqN': '392', '2UT3Ips': '393', 'COkaPpN': '394', 'mrjFNGX': '395', '9dArUsU': '396', 'NsCDfKJ': '397', 'oRz3Ms8': '398', '0WBXRH9': '399', '55A2AUa': '400', 'a1F6RKt': '401', 'GaQNDn2': '402', 'bspb4sM': '403', 'ET5rdX2': '404', 'OH0Nuxb': '405', 'QHHMKB3': '406', 'QebeqSh': '407', '9A372Gl': '408', '95L0GB6': '409', 'yiS7xVy': '410', 'iutC6Qy': '411', 'aL1hjkE': '412', 'qv6y9xq': '413', '8NMWu57': '414', 'NssAP3V': '415', 'x76JHsa': '416', '1f2zSW8': '417', 'qERLZTk': '418', 'hV8hDZV': '419', 'ZbqRtuL': '420', 'nHevBnI': '421', 'T4u74mL': '422', 'AQFwrJx': '423', 'ZWajHQq': '424', 'CaqKlOXWIAAvy8L': '425', '91sHVxQ': '426', '7S6NEVD': '427', 'meTteYg': '428', 'PdPdKfd': '429', 'VNZBamf': '430', 'QLQ4taO': '431', 'HSVyqdL': '432', 'lVOjKnR': '433', 'gOHWdbe': '434', 'MmdEDKF': '435', 'i1KC0aU': '436', 'h6Pkqkr': '437', 'TOuTChz': '438', 'mVjzFQY': '439', 'WiqI5X3': '440', 'iZgFwkB': '441', 'h2nKIqJ': '442', '88pU7QL': '443', '0EwB4LT': '444', '9BmlvFy': '445', '0vqvDs5': '446', '3Zz0vux': '447', 'BWopGu8': '448', 'HUU1Fay': '449', 'CIFrUHV': '450', '6faN492': '451', 'TgSYoWe': '452', 'b3IhU9Z': '453', 'VvMdZpV': '454', 'VoNgxAD': '455', '0P5i3yI': '456', 'CfUTJmj': '457', '2Qo03El': '458', 'Lo2ylA3': '459', 'zAUFD7y': '460', 'XTg6oYN': '461', 'AGGHufY': '462', 'D9Zr2dU': '463', 'QnLfqfX': '464', 'sKiuQVf': '465', 'T2Jt2ry': '466', 'D40IF3q': '467', 'CdsdCaHXIAEw1An': '468', 'akXZYL5': '469', 'Q29wRyg': '470', 'm85k7GR': '471', '7wKCNn2': '472', 'kO0gHF2': '473', '05nBEvh': '474', 'lR6IYpG': '475', 'PpxbGZC': '476', 'Vi4Kjyx': '477', 'TjsILvp': '478', 'KiYSMvi': '479', 'fW7CAvO': '480', 'LE4FyLF': '481', 'mSA9fTA': '482', 'ayNSttp': '483', 'SZTu5uv': '484', 'CySpKqI': '485', 'CibcteKWsAEgqh0': '486', '5vVB395': '487', 'st6NozD': '488', 'AthUmaI': '489', 'qSyop93': '490', 'XoR7xtd': '491', 'X2dmDrp': '492', 'VYZEg9A': '493', 'ykaOeQZ': '494', 'xJiXDnY': '495', 'XYERAZg': '496', 'N4EIWJX': '497', '0SvkQMd': '498', '8Fwa0DR': '499', 'jkdwXzP': '500', 'sG8faWv': '501', 'FvlpogZ': '502', '7eTNqFK': '503', '0bOKK62': '504', 'mYU8r9Y': '505', 'qnRUXr8': '506', 'JYTdhl9': '507', 'hOxu3fK': '508', '6jA3QAb': '509', 'yGHKtzg': '510', 'zl617iP': '511', 'WcXzDk2': '512', 'zoIImWJ': '513', 'u9FeEuE': '514', 'hoApJnq': '515', 'L6WFxJs': '516', 'qoZxIKO': '517', 'LJ3r8Gy': '518', 'C0JFBbU': '519', '9kZb6UD': '520', 'DDWmGxP': '521', 'BmKzn6gIYAIwC-U': '522', 'dOr1Ljj': '523', 'l7UFL6x': '524', 'oqt51kd': '525', 'iAztwx4': '526', '0p7HM8n': '527', '8jiJ4Zt': '528', '5hhUFqA': '529', 'C5mm2gl': '530', 'RX8t7Jd': '531', 'y898T2X': '532', 'ZffTHk2': '533', '4LoGRB5': '534', '5aomAQA': '535', 'Cc8PZCkXIAAmG0f': '536', 'oKD3FG3': '537', 'Hf65xSJ': '538', 'KIiW0Lw': '539', 'ugTPNMd': '540', 'dZVtSfm': '541', 'jAi3iI1': '542', 'V6MP5lX': '543', '4KXGjz0': '544', 'Cm96pUGVMAA7HQz': '545', '9HHSnfe': '546', 'Okb6dJB': '547', 'zr5VeQM': '548', 's9Z2Han': '549', '9n1Tjiz': '550', 'NDFCXLG': '551', 'l0P4MzC': '552', 'ArmX82G': '553', 'pEPz8iF': '554', 'sbWUiil': '555', 'wK9e9jg': '556', 'zOImoh5': '557', 'BCah7Wz': '558', 'RbOxrFZ': '559', 'uDbb5Gk': '560', 'MZQc8jz': '561', '4j07SmD': '562', 'Nbo6iZU': '563', '8Djb33H': '564', 'nEE5sjy': '565', 'Zh4CZSe': '566', 'eI2N5iQ': '567', 'WDUgdRk': '568', 'bDIJ7sx': '569', 'hF5Bi0f': '570', 'JbWv3sM': '571', 'XniGGU5': '572', 'CwHWOROW8AA09nd': '573', 'zZIOwqg': '574', 'VOiuzA6': '575', '0Te3Qys': '576', 'L7jzD7i': '577', 'fsgozVS': '578', 'W5DXy2F': '579', 'BVzw01E': '580', 'QLgpXMT': '581', '0UQh5Eo': '582', 'eMSADPX': '583', 'nefjbtQ': '584', '4q1tono': '585', '2aiRLNR': '586', 'fyAh3I0': '587', '2NS27kx': '588', 'iQ0COdD': '589', 'Cco_BiaUMAAla6m': '590', 'mZvaiMI': '591', 'xYTrJJp': '592', 'sW3bJWW': '593', 'T35mICA': '594', 'wKSn3TS': '595', '6WELqLH': '596', 'CvOg2va': '597', 'iyTe9SI': '598', 'wgVTz9g': '599', '3fsUJvp': '600', 'WSdFOAD': '601', 'JBUKM8i': '602', 'Ce0OU-iUIAALkmn': '603', 'WyNvTye': '604', 'gbt5ETH': '605', 'bpc3wap': '606', '5oT4c6d': '607', 'XMReWCt': '608', 'Ob1Q9v8': '609', 'D65y7fq': '610', 'ZEjNdo0': '611', 'DLr1Uo7': '612', 'jQ6NQqx': '613', 'LAPqPzX': '614', 'ZMaAKc9': '615', 'aXJH7EU': '616', 'gK5tKHN': '617', '2tuVUzz': '618', 'QNB8H2s': '619', 'SXvE9sL': '620', 'zRgnrbT': '621', 'uluP5WP': '622', '6V4OrKm': '623', 'nsEOOZO': '624', 'bUfWNUp': '625', 'zPT9dFf': '626', 'B_LgxAvVIAA_lu8': '627', 'Cf87U8rW4AAtlvE': '628', 'ewKXYvk': '629', 'AMzup5d': '630', 'ztdf8dZ': '631', 'HyQBPvH': '632', 'TyYDiSx': '633', 'BXs0Wy8': '634', 'QmLaggk': '635', 'fmTCPZO': '636', 'BVzBIgy': '637', '1DY6I9Q': '638', 'ExlQpoj': '639', 'vsFSedb': '640', 'aKBYqSs': '641', 'xGBX7zN': '642', '8ItzKTN': '643', 'Bq3Oe2TIAAAUgik': '644', 'Wl32YJG': '645', '2TAk7vi': '646', 'LW5TKwe': '647', 'f0kDhU7': '648', 'WVeMKB9': '649', 'zvdzU3y': '650', '3cvbmM0': '651', 'OA7aCgM': '652', 'WZ1NLSt': '653', 'JT26Var': '654', 'Im9iERE': '655', 'gzWfYg3': '656', 'yBjFq4v': '657', 'wzOK4S3': '658', 'YW7mx62': '659', 'm0FKJHM': '660', 'V7Ie6Os': '661', 'so4V8X7': '662', 'otfhxeo': '663', 'd5ZqKJ5': '664', '6uTXEgh': '665', 'qZ8nvyF': '666', 'H6vd1DD': '667', 'cnkpnzp': '668', '4hxCAfW': '669', 'tMzNcad': '670', 'H0Bh7VP': '671', 'Rqzw9Co': '672', 'Nyo6x2Y': '673', 'Tf1mhKt': '674', 'pdaFFC4': '675', '5kSMRv3': '676', 'we4hhWi': '677', 'X6BtuWV': '678', 'C2Rjlth': '679', 'VOFh4qX': '680', '7COAAL3': '681', 'NyP2PbK': '682', 'whhrfSE': '683', 'qHkGmLt': '684', 'fdZpYpb': '685', 'ruSN6zI': '686', 'xyi7yMK': '687', '3bqSgzm': '688', 'QQnZqPB': '689', 'hLMgzsp': '690', 'H7Ob55y': '691', 'rDy2M4O': '692', 'IpwuEVr': '693', 'orV6OVk': '694', '5Oznxbq': '695', '1BeUIMs': '696', 'DheGAH3': '697', 'dCBEORM': '698', 'rmlkOMu': '699', 'DNwiCOI': '700', 'CcFmlzLW8AAxHkM': '701', 'eRUFvRF': '702', 'xQkfJse': '703', '5NWPOQp': '704', 'jG51M2l': '705', 'XX28NwO': '706', 'fLxsrjJ': '707', 'IWXLlFJ': '708', 'nqfekut': '709', 'JHF6hFI': '710', 's1qivuu': '711', 'wwEGRrn': '712', 'TqFVPVE': '713', 'k129f29': '714', 'zs6maXI': '715', 'LkX8Ei3': '716', '6NFcW7v': '717', 'PjZa9M1': '718', 'lPguwLG': '719', '16vTnwg': '720', 'BNPocHU': '721', '209BoJK': '722', '2pt7Wkd': '723', 'JYTlRZK': '724', 'Lsjt4TJ': '725', 'Ci45dogWEAA2bSb': '726', 'TWjXbPK': '727', 'ysAk127': '728', 'cAm7azy': '729', 'pvCTAHi': '730', 'ky2vvAw': '731', 'tcXzxyV': '732', 'fC5euc2': '733', 'koxcr8J': '734', 'JK1w4yI': '735', 'FSeqQG9': '736', 'h92g9FL': '737', 'yWd3NEB': '738', 'XqB2jSY': '739', 'EMshcj9': '740', 'aK7Mzo4': '741', 'LeHuK4y': '742', 'CkbzAqAUoAUJ311': '743', '7Lg5Rd2': '744', 'VAJZJJV': '745', 'CecBrceWQAQR19j': '746', 'rZGXxM4': '747', 'WDSz8Sa': '748'}




# with open('/workspace/MultiOFF_Dataset/test.json','r',encoding='utf8')as fp:
#     for line in fp.readlines():
#         dic = json.loads(line)
#         print(dic)


# with open('/workspace/MultiOFF_Dataset/train.json','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         for i in dic:
#             if i["image_name"] == "image_name":
#                 continue
#             else :
#                 dicc = {}
#                 dicc["image"] = i["image_name"].replace(".png", "")
#                 dicc["sentence"] = i["sentence"]
#                 dicc["label"] = i["label"]
#                 if dicc["label"] == "offensive":
#                     dicc["label"] = 1
#                 else :
#                     dicc["label"] = 0
#                 a.append(dicc)
#     with open('/workspace/MultiOFF_Dataset/ALBEF/train.json','a',encoding='utf8')as fp:
#         json.dump(a, fp)


# with open('/workspace/multioff/ALBEF/test.json','r',encoding='utf8')as fp:
#     a = []
#     for line in fp.readlines():
#         dic = json.loads(line)
#         for i in dic:
#             dicc = {}
#             dicc["image"] = i["image"].replace(".jpg", "")
#             dicc["sentence"] = i["sentence"]
#             dicc["label"] = i["label"]
#             a.append(dicc)
#     with open('/workspace/multioff/ALBEF/test1.json','a',encoding='utf8')as fp:
#         json.dump(a, fp)



with open('/workspace/Harm_P/train.jsonl','r',encoding='utf8')as fp:
    a = []
    for line in fp.readlines():
        dic = json.loads(line)
        dicc = {}
        dicc["image"] = dic["id"]
        dicc["sentence"] = dic["text"]
        if dic["labels"][0]=="not harmful":   
            dicc["label"] = 0
        else :
            dicc["label"] = 1
        a.append(dicc)
    with open('/workspace/Harm_P/ALBEF/train.json','a',encoding='utf8')as fp:
        json.dump(a, fp)





