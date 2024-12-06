import sys
sys.path.append('.')
sys.path.append('src')
from tmi_logic import TMIAnalysis
import pandas as pd

def add_tmi_to_data(path='', delimiter=',', cols=None, corenlp_url=''):
    sep = delimiter
    if cols is not None:
        data = pd.read_csv(path, sep=sep, names=cols)
    else:
        data = pd.read_csv(path, sep=sep)
    data_df = data.iloc[23000:53804]
    print(data_df.head(5))
    tmi_obj = TMIAnalysis(df=data_df, corenlp_url=corenlp_url)
    data_df = tmi_obj.is_biased_from_descriptor()
    print("Done adding descriptor class\n", data_df.head(5), "\n")
    path_split = path.split('/')
    out_path = '/'.join(path_split[:-1]) + '/tmi_' + path_split[-1]
    print("Out path......", out_path, "\n")
    data_df.to_csv(out_path, sep=sep, index=False, header=True)


if __name__ == '__main__':
    column_names = ["revid", "src_tok", "tgt_tok", "headline",
                    "tgt_raw", "src_pos_tags", "targe_parse_tags"]
    add_tmi_to_data('src/bias_tagger_detection/bias_data/WNC/biased.word.train',
                    delimiter='\t', cols=column_names, corenlp_url="http://cccxc438.pok.ibm.com:9000/")
    # add_tmi_to_data(sys.argv[1], corenlp_url=sys.argv[2])
