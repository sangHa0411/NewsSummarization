
import re

class Preprocessor :
    def __init__(self,) :
        self.base_block = re.compile(r'(\\n|\n)')
        self.unk_block = re.compile('[\u3000-\u303f\ud800—\udbff\ue000—\uf8ff]') # unk tokens
        self.cjk_block = re.compile('[\u2e80—\u2eff\u3400—\u4dbf\u4e00—\u9fff]') # chinese , japanese, korean
        self.outrange_block = re.compile('[\uffff-\U000e007f]')                  # outrange

    def preprocess4train(self, dataset) :
        assert isinstance(dataset, dict)
        document = dataset['document']
        summary = dataset['summary']

        document = self.sen_preprocess(document)
        summary = self.sen_preprocess(summary)

        dataset['document'] = document
        dataset['summary'] = summary
        return dataset

    def preprocess4test(self, dataset) :
        assert isinstance(dataset, dict)
        document = dataset['document']
        dataset['document'] = self.sen_preprocess(document)
        return dataset

    def sen_preprocess(self, sen) :
        sen = self.base_block.sub(' ', sen)
        sen = self.unk_block.sub(' ', sen)
        sen = self.cjk_block.sub(' ', sen)
        sen = self.outrange_block.sub(' ', sen)
        sen = re.sub('\s+' , ' ' , sen)
        return sen
