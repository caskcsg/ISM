def get_temps(tokenizer):
    temps = {}
    with open('/workspace/project/ALBEF/temp.txt', 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            info = {}
            info['id'] = items[0].strip()
            info['template'] = items[1:-1]  + [tokenizer.mask_token]#mask_token="[MASK]"
            info['label_name'] = items[-1]
            temps[info['id']] = info                                                
            #print(info)
#{'name': '0', 'temp': ['it', 'is', '<mask>'], 'label': 'terrible'}
#{'name': '1', 'temp': ['it', 'is', '<mask>'], 'label': 'bad'}
#{'name': '2', 'temp': ['it', 'is', '<mask>'], 'label': 'common'}
#{'name': '3', 'temp': ['it', 'is', '<mask>'], 'label': 'good'}
#{'name': '4', 'temp': ['it', 'is', '<mask>'], 'label': 'nice'}
    #print(temps) #自动去重
            print(info)
    return temps


