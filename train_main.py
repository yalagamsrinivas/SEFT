from torch.utils.data import DataLoader
import numpy as np
import torch
import json
import warnings
import sys
import os
from collections import defaultdict
from tqdm import tqdm

sys.path.append('.')

from model.config_loader import config, path
from model.data_loader import StockData
if config['model']['name'] == 'seft_v1':
    from model.seft_v1 import SEFT, SEFTNEWWrapper
if config['model']['name'] == 'seft_v2':
    from model.seft_v2 import SEFT, SEFTNEWWrapper
if config['model']['name'] == 'seft_v3':
    from model.seft_v3 import SEFT, SEFTNEWWrapper
if config['model']['name'] == 'seft_v4':
    from model.seft_v4 import SEFT, SEFTNEWWrapper
if config['model']['name'] == 'seft_v5':
    from model.seft_v5 import SEFT, SEFTNEWWrapper

from model.GloveEmbedding import glove_embedding
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.double)

np.random.seed(42)

device = config['model']['device']
emb_dev = config['model']['device']

experiment_name = '{}_{}_{}_nmsg_{}_nword_{}_lr_{}_heads_{}_hidden_dim_{}_num_layers_{}'.format(
    config['model']['prefix'], config['model']['name'], config['stage'],
    config['model']['max_n_msgs'], config['model']['max_n_words'], config['train']['lr'],
    config['model']['num_heads'], config['model']['text_hidden_dim'], config['model']['num_layers']
)

print('-- RUNNING EXPERIENMENT: {} --'.format(experiment_name))

batch_size=config['model']['batch_size']

embedding = glove_embedding(path=path.glove, embedding_dim=config['model']['word_embed_size'],
                            vocab_list=config['vocab'])
embedding.to(device)
network = SEFT(embedding=embedding, embedding_dim=config['model']['word_embed_size'], num_heads=config['model']['num_heads'], num_layers=config['model']['num_layers'], hidden_dim=config['model']['text_hidden_dim'],
               technical_feature_dim=config['model']['tech_feature_size'],output_dim=config['model']['output_dim'],device=device)
model = SEFTNEWWrapper(experiment_name, network, 'Adam', 1, config)

model.setup(True)
    
tickers = []
for indu in config['stocks'].values():
    tickers += indu
param_norm = defaultdict(list)

# log and trajectory
if not os.path.exists('./log/train/{}'.format(experiment_name)):
    os.mkdir('./log/train/{}'.format(experiment_name))
if not config['train']['resume']:   # start training from scratch
    start_epoch = 0
else:
    start_epoch = model.load_model(ckpt_path='./checkpoints/', epoch=None, load_optimizer=True)

print('finished initializing model...')    

if config['stage'] == 'train':
    
    with open('./log/train/{}/config.txt'.format(experiment_name), 'w') as f:
        f.write(json.dumps({'model':config['model'], 'dates':config['dates'], 
                            'stocks': config['stocks'], 'train':config['train']}))
    log = open('./log/train/{}/log.txt'.format(experiment_name), 'a')
    log.flush()

print('{} on :'.format(config['stage']))
print(tickers)

if config['stage'] == 'train':
    
    # load train dataset
    train_dset = StockData(config['dates']['train'], tickers, device)
    train_loader = DataLoader(train_dset, batch_size, shuffle=True)
    print('training datset for dates {} containing {} instance'.format(config['dates']['train'], len(train_dset)))
    
    inputs= next(iter(train_loader))
    
# load test dataset
test_dset = StockData(config['dates']['test'], tickers, device)
test_loader = DataLoader(test_dset, 64, shuffle=False)
print('test datset for dates {} containing {} instance'.format(config['dates']['test'], len(test_dset)))

# training phase
if config['stage'] == 'train':

    for epoch in tqdm(range(start_epoch, config['train']['epochs'])):

        train_acc, train_f1, train_pos, train_mles, train_precision, train_recall, train_mcc = model.run(train_loader,
                                                                                                         1,
                                                                                                         interval=5000)
        test_acc, test_f1, test_pos, test_mles, test_precision, test_recall, test_mcc = model.run(test_loader, 0,
                                                                                                  interval=10)
        report = '| epoch:{} | train_acc:{:.4f},train_f1:{:.4f}, train_pos_rate:{:.3f}, train mle: {:.4f}, train_precision: {:.4f}, train_recall: {:.4f}, train_mcc: {:.4f}' \
                 '    | test_acc:{:.3f}, test_f1:{:.3f}, test_pos:{:.3f}, test mle: {:.4F}, test_precision: {:.4F}, test_recall: {:.4F}, test_mcc: {:.4F}'.format(
            epoch,
            train_acc, train_f1, train_pos, np.mean(train_mles), train_precision, train_recall, train_mcc, test_acc,
            test_f1, test_pos, np.mean(test_mles), test_precision, test_recall, test_mcc)
        print(report)
        log.write(report + '\n')
        log.flush()

        if epoch % 1 == 0 and epoch >= 0:
            model.save_model('./checkpoints'.format(experiment_name), epoch, save_optimizer=True)

    # save the results
    log.close()

if config['stage'] == 'test':
    epoch = 9
    model.load_model(ckpt_path='./checkpoints/', epoch=epoch, load_optimizer=True)
    acc, f1, pos, mle, precision, recall, mcc = model.run(test_loader, 0, interval=500)
    print(
        'test ----->   checkpoint: {}, acc: {}, f1: {}, pos rate: {},mle: {}, precision: {}, recall: {}, mcc: {}'.format(
            epoch, acc, f1, pos, mle, precision, recall, mcc))