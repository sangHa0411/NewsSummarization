
import os
import json
import pandas as pd
import collections
import random
from tqdm import tqdm
from datasets import Dataset, DatasetDict


class ArticleLoader :
    def __init__(self, text_dir_path, info_file_path, max_doc_len, min_doc_len) :
        self.text_dir_path = text_dir_path
        self.info_file_path = info_file_path
        self.max_doc_len = max_doc_len
        self.min_doc_len = min_doc_len

    def load_data(self) :
        train_path = os.path.join(self.text_dir_path, 'train_data.json')
        validation_path = os.path.join(self.text_dir_path, 'validation_data.json')

        if (os.path.isfile(train_path) and os.path.isfile(validation_path)) == False :
            self.save_data()

        train_df = pd.read_json(train_path)
        validation_df = pd.read_json(validation_path)

        dataset = DatasetDict({'train' :Dataset.from_pandas(train_df) ,
            'validation': Dataset.from_pandas(validation_df)}
        )
        
        return dataset

    def save_data(self) :
        df = pd.read_csv(self.info_file_path)
        df = df[['title','date','category','text']]

        data_size = len(df)
        index_map = collections.defaultdict(list)
        for i in range(data_size) :
           category =  df.iloc[i]['category']
           index_map[category].append(i)

        train_data = []
        val_data = []

        for label in index_map.keys() :
            idx_list = index_map[label]

            val_size = int(len(idx_list) * 0.2)
            val_index = random.sample(idx_list, val_size)
            train_index = list(set(idx_list) - set(val_index))

            train_data.extend(train_index)
            val_data.extend(val_index)

        random.shuffle(train_data)
        random.shuffle(val_data)

        print('Save Train Data')
        self.to_json(train_data, df, 'train_data')
        print('Save Validation Data')
        self.to_json(val_data, df, 'validation_data')

    def load_text(self, file_path) :
        f = open(file_path, 'r')
        text = f.read()

        passages = text.split('\n')
        passages = [p for p in passages if p != '']
        text = ' '.join(passages[:5])
        return text


    def to_json(self, index, data, name) :
        df_data = []
        for i in tqdm(index) :
            try : 
                title, text_path = data.iloc[i][['title', 'text']]
                title = title.split(' | ')[0]
                text = self.load_text(text_path)

                if len(text) >= self.min_doc_len and len(text) <= self.max_doc_len:
                    dict_data = {'summary' : title, 'document' : text}
                    df_data.append(dict_data)
            except :
                continue

        print('Data Size : %d' %len(df_data))
        df_data = pd.DataFrame(df_data)
        df_data.to_json(os.path.join(self.text_dir_path, name+'.json'))

