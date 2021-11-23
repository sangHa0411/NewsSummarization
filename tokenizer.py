import os
import json
import pandas as pd
from transformers import T5TokenizerFast, AutoTokenizer

class TokenizerOptimization :
    def __init__(self, tokenizer, dir_path, extra_token_df) :
        assert isinstance(tokenizer, T5TokenizerFast) and isinstance(extra_token_df, pd.DataFrame)
        self.tokenizer = tokenizer
        self.size = self.get_size(tokenizer)
        self.extra_vocab_list = list(extra_token_df['Token'][:self.size])
        self.dir_path = dir_path

    def get_size(self, tokenizer) :
        count = 0 
        for tok in tokenizer.vocab :
            if 'extra_id' in tok :
                count += 1
        return count

    def optimize_config(self, extra_vocab_list) :
        config_path = os.path.join(self.dir_path, 'tokenizer_config.json')
        with open(config_path) as json_file:
            tokenizer_config = json.load(json_file)

        tokenizer_config['additional_special_tokens'] = extra_vocab_list
        with open(config_path, 'w') as json_file:
            json.dump(tokenizer_config, json_file)

    def optimize_tokens_map(self, extra_vocab_list) :
        tokens_map_path = os.path.join(self.dir_path, 'special_tokens_map.json')
        with open(tokens_map_path) as json_file:
            tokenizer_config = json.load(json_file)

        tokenizer_config['additional_special_tokens'] = extra_vocab_list
        with open(tokens_map_path, 'w') as json_file:
            json.dump(tokenizer_config, json_file)

    def optimize_tokenizer(self, extra_vocab_list) :
        tokenizer_path = os.path.join(self.dir_path, 'tokenizer.json')
        with open(tokenizer_path) as json_file:
            tokenizer_data = json.load(json_file)

        extra_size = len(extra_vocab_list)
        start_point = len(self.tokenizer) - extra_size
        for i in range(start_point, len(self.tokenizer)) :
            tokenizer_data['model']['vocab'][i] = [extra_vocab_list[i-start_point], 0.0]

        start_point = len(tokenizer_data['added_tokens']) - extra_size
        for i in range(start_point, len(tokenizer_data['added_tokens'])) :
            tokenizer_data['added_tokens'][i]['content'] = extra_vocab_list[i-start_point]

        with open(tokenizer_path, 'w') as json_file:
            json.dump(tokenizer_data, json_file)

    def optimize(self) :
        self.tokenizer.save_pretrained(self.dir_path)
        self.optimize_tokens_map(self.extra_vocab_list)
        self.optimize_config(self.extra_vocab_list)
        self.optimize_tokenizer(self.extra_vocab_list)      

        tokenizer = AutoTokenizer.from_pretrained(self.dir_path, extra_ids=0)
        return tokenizer


