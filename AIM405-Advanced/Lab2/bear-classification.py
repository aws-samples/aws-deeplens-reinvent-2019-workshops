import mxnet as mx
import numpy as np
import os, time, shutil
os.system("pip install gluoncv==0.5.0")

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model

import argparse
import json
import sys

import logging
logging.basicConfig(level=logging.INFO)

def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()
    
if __name__ == '__main__':

    # Receive hyperparameters passed via create-training-job API
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    
    args = parser.parse_args()
    
    model_dir = args.model_dir
    
    # class: brown, polar, no bear
    classes = 3
    epochs = args.epochs
    lr = 0.001
    per_device_batch_size = args.batch_size
    num_workers = 8
    
    try:
        num_cpus = int(os.environ['SM_NUM_CPUS'])
        num_gpus = int(os.environ['SM_NUM_GPUS'])
    except KeyError:
        num_gpus = 0
    
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    batch_size = per_device_batch_size * max(num_gpus, 1)
    #ctx = mx.gpu() if num_gpus > 0 else mx.cpu()

    jitter_param = 0.4
    lighting_param = 0.1
    model_name = 'MobileNet1.0'
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, 
                                 contrast=jitter_param,
                                 saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ])
    
    train_path = os.path.join(args.train, 'train')
    val_path = os.path.join(args.train, 'val')
    test_path = os.path.join(args.train, 'test')

    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)
    
    test_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    
    finetune_net = get_model(model_name, pretrained=True)

    with finetune_net.name_scope():
        finetune_net.output = nn.Dense(classes)
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    trainer = gluon.Trainer(finetune_net.collect_params(), 'adam', 
                        {'learning_rate': lr})

    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()

    num_batch = len(train_data)
    
    # Training Loop 
    for epoch in range(epochs):
        tic = time.time()
        train_loss = 0
        metric.reset()

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            metric.update(label, outputs)

        _, train_acc = metric.get()
        train_loss /= num_batch

        _, val_acc = test(finetune_net, val_data, ctx)

        logging.info('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
                 (epoch, train_acc, train_loss, val_acc, time.time() - tic))

    _, test_acc = test(finetune_net, test_data, ctx)
    logging.info('[Finished] Test-acc: %.3f' % (test_acc))
    
    save_model_name = model_dir + '/mobilenet1.0-bear'

    finetune_net.export(save_model_name)
    net_with_softmax = finetune_net(mx.sym.var('data'))
    net_with_softmax = mx.sym.SoftmaxOutput(data=net_with_softmax, name='softmax')
    net_with_softmax.save('{}-symbol.json'.format(save_model_name))

    
# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def load_model(s_fname, p_fname):
    """
    Load model checkpoint from file.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    symbol = mx.symbol.load(s_fname)
    save_dict = mx.nd.load(p_fname)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params   
        
def model_fn(model_dir):
    """
    Load the model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model 
    """
    
    model_symbol = '{}/mobilenet1.0-bear-symbol.json'.format(model_dir)
    model_params = '{}/mobilenet1.0-bear-0000.params'.format(model_dir)
    
    logging.info('[model_symbol] {}'.format(model_symbol))
    logging.info('[model_params] {}'.format(model_params))
    
    sym, arg_params, aux_params = load_model(model_symbol, model_params)
    DSHAPE = (1,3,224,224)
    # The shape of input image. 
    dshape = [('data', DSHAPE)]

    ctx = mx.cpu() # USE CPU to predict... 
    net = mx.mod.Module(symbol=sym,context=ctx)
    net.bind(for_training=False, data_shapes=dshape)
    net.set_params(arg_params, aux_params)
    
    return net 
