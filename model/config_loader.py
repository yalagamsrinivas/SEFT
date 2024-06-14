import logging
import logging.config
import os
import io, re
import json
import sys
import warnings
import argparse
sys.path.append("..") 

warnings.filterwarnings("ignore")

def parse_config():

    parser = argparse.ArgumentParser()
    
    # path argument
    parser.add_argument('--paths_data', type=str, default='stocknet_dataset/', help='relative data path')
    parser.add_argument('--paths_text', type=str, default='text/tweet', help='relative text path')
    parser.add_argument('--paths_label', type=str, default='label/return', help='relative label path')
    parser.add_argument('--paths_technical', type=str, default='technical', help='relatice technical feature path')
    parser.add_argument('--paths_resource', type=str, default='./resource', help='absolute resource path')
    parser.add_argument('--paths_glove', type=str, default='glove.twitter.27B.50d.txt', help='glove file name')
    parser.add_argument('--paths_vocab_tweet', type=str, default='extended_customized_vocab_fin_tweet.txt', help='vocal file name')
    parser.add_argument('--paths_checkpoints', type=str, default='checkpoints', help='relative checkpoints folder path')
    parser.add_argument('--paths_log', type=str, default='log/', help='relative log folder')
    
    # model argument 
    parser.add_argument('--model_name', type=str, default='seft_simple', help='name of model to be used')
    parser.add_argument('--model_word_embed_type', type=str, default='glove', help='word embedding type')
    parser.add_argument('--model_weight_init', type=str, default='xavier_uniform', help='weight initialiation method')
    
    parser.add_argument('--model_word_embed_size', type=int, default=50, help='embedding size')
    #parser.add_argument('--model_stock_embed_size', type=int, default=150, help='')
    parser.add_argument('--model_tech_feature_size', type=int, default=3, help='')
    parser.add_argument('--model_num_heads', type=int, default=2, help='')
    parser.add_argument('--model_vocab_size', type=int, default=79674, help='')
    parser.add_argument('--model_text_hidden_dim', type=int, default=20, help='')
    parser.add_argument('--model_output_dim', type=int, default=2, help='')
    parser.add_argument('--model_batch_size', type=int, default=1, help='')
    parser.add_argument('--model_num_layers', type=int, default=20, help='')
    parser.add_argument('--model_init_stock_with_word', type=int, default=0, help='')
    parser.add_argument('--model_device', type=str, default='cuda')

    parser.add_argument('--model_day_step', type=int, default=5, help='')
    parser.add_argument('--model_day_shuffle', type=int, default=0, help='')
    parser.add_argument('--model_ticker_combine', type=int, default=0, help='')
    parser.add_argument('--model_ticker_shuffle', type=int, default=1, help='')
    parser.add_argument('--model_text_combine', type=int, default=0, help='')
    parser.add_argument('--model_max_n_msgs', type=int, default=5, help='')
    parser.add_argument('--model_max_n_words', type=int, default=40, help='')
    parser.add_argument('--model_threshold', type=int, default=1, help='')
    
    parser.add_argument('--model_prefix', type=str, default='ljmu', help='')
    
    # train argument
    parser.add_argument('--train_lr', type=float, default=0.001, help='')
    parser.add_argument('--train_epsilon', type=int, default=1, help='')
    parser.add_argument('--train_epochs', type=int, default=50, help='')
    parser.add_argument('--train_min_date_ratio', type=int, default=0, help='')
    parser.add_argument('--train_weight_decay', type=float, default=0.01, help='')
    parser.add_argument('--train_schedule_step', type=int, default=5, help='')
    parser.add_argument('--train_schedule_gamma', type=float, default=0.5, help='')
    parser.add_argument('--train_resume', type=int, default=0, help='')
     
    # stocks argument
    parser.add_argument('--stocks_materials', type=list, default=['XOM', 'RDS-B', 'PTR', 'CVX', 'TOT', 'BP', 'BHP', 'SNP', 'SLB', 'BBL'], help='')
    parser.add_argument('--stocks_consumer_goods', type=list, default=['PG', 'BUD', 'KO', 'PM', 'TM', 'PEP', 'UN', 'UL', 'MO'], help='')
    parser.add_argument('--stocks_healthcare', type=list, default=['JNJ', 'PFE', 'NVS', 'UNH', 'MRK', 'AMGN', 'MDT', 'ABBV', 'SNY', 'CELG'], help='')
    parser.add_argument('--stocks_services', type=list, default=['AMZN', 'BABA', 'WMT', 'CMCSA', 'HD', 'DIS', 'MCD', 'CHTR', 'UPS', 'PCLN'], help='')
    parser.add_argument('--stocks_utilities', type=list, default=['NEE', 'DUK', 'D', 'SO', 'NGG', 'AEP', 'PCG', 'EXC', 'SRE', 'PPL'], help='')
    parser.add_argument('--stocks_cong', type=list, default=['IEP', 'HRG', 'CODI', 'REX', 'SPLP', 'PICO', 'AGFS', 'GMRE'], help='')
    parser.add_argument('--stocks_finance', type=list, default=['BCH', 'BSAC', 'BRK-A', 'JPM', 'WFC', 'BAC', 'V', 'C', 'HSBC', 'MA'], help='')
    parser.add_argument('--stocks_industrial_goods', type=list, default=['GE', 'MMM', 'BA', 'HON', 'UTX', 'LMT', 'CAT', 'GD', 'DHR', 'ABB'], help='')
    parser.add_argument('--stocks_tech', type=list, default=['GOOG', 'MSFT', 'FB', 'T', 'CHL', 'ORCL', 'TSM', 'VZ', 'INTC', 'CSCO'], help='')
    
    parser.add_argument('--dates_train', type=str, 
                        default=[
                            ['2014-01-01', '2015-09-30']],
                        help='')
    parser.add_argument('--dates_test', type=list,
                        default=[
                            ['2015-10-01', '2015-12-31']], help='')
    
    # stage argument
    parser.add_argument('--stage', type=str, default='train', help='')

    # parse argument
    args = parser.parse_args()
    
    category = ['paths', 'model', 'train', 'stocks', 'dates', 'pgd']
    config = {}
    for c in category:
        regex = '^{}_'.format(c)
        config[c] = {}
        length = len(c)
        for arg in args.__dict__.keys():
            if re.findall(regex, arg):
                config[c][arg[length+1:]] = args.__dict__[arg]
    config['stage'] = args.stage


    return config
class PathParser:

    def __init__(self, config_path):
        self.root = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir))
        self.log = os.path.join(self.root, config_path['log'])
        self.data = os.path.join(self.root, config_path['data'])
        self.res = config_path['resource']
        self.checkpoints = os.path.join(self.root, config_path['checkpoints'])

        self.glove = os.path.join(self.res, config_path['glove'])

        self.technical = os.path.join(self.data, config_path['technical'])
        self.text = os.path.join(self.data, config_path['text'])
        self.label = os.path.join(self.data, config_path['label'])
        self.vocab = os.path.join(self.res, config_path['vocab_tweet'])

config = parse_config()  
path = PathParser(config_path=config['paths'])

with io.open(str(path.vocab), 'r', encoding='utf-8') as vocab_f:
    config['vocab'] = json.load(vocab_f)
    config['vocab_size'] = len(config['vocab']) + 1  # for UNK

# logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
log_fp = os.path.join(path.log, '{0}.log'.format('model'))
file_handler = logging.FileHandler(log_fp)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

if __name__ == '__main__':
    parse_config()