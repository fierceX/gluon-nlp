import json
import gluonnlp as nlp

import mxnet as mx
from mxnet import gluon

from mxnet.gluon import Block
from mxnet.gluon import nn
from mxnet import nd

import collections
import numpy as np
import json

from tqdm import tqdm
import six

from bert import BERTSquad, BERTloss
from dataset import SQData, SQuADTransform
from evaluate import predictions


ctx = mx.gpu(0)
epochs = 2
lr = 3e-5

batch_size = 12

bert, vocab = nlp.model.bert_12_768_12(dataset_name='book_corpus_wiki_en_uncased',
                                       pretrained=True, ctx=ctx, use_pooler=False,
                                       use_decoder=False, use_classifier=False,)

berttoken = nlp.data.BERTTokenizer(vocab=vocab)


train_data = SQData('./squad1.1/train-v1.1.json')

dev_data = SQData('./squad1.1/dev-v1.1.json', is_training=False)


train_data = train_data.transform(
    SQuADTransform(berttoken, max_seq_length=384))

dev_data = dev_data.transform(SQuADTransform(
    berttoken, max_seq_length=384, is_training=False)._transform)

train_dataloader = mx.gluon.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)

dev_dataloader = mx.gluon.data.SimpleDataset(dev_data)


net = BERTSquad(bert=bert)
net.Dense.initialize(init=mx.init.Normal(0.02), ctx=ctx)
net.hybridize(static_alloc=True)

loss_function = BERTloss()
loss_function.hybridize(static_alloc=True)


trainer = gluon.Trainer(net.collect_params(), 'adam', {
                        'learning_rate': lr, 'epsilon': 1e-9})


num_train_examples = len(train_data)
num_train_steps = int(num_train_examples / batch_size * epochs)
warmup_ratio = 0.1
num_warmup_steps = int(num_train_steps * warmup_ratio)
step_num = 0
differentiable_params = []

for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
    v.wd_mult = 0.0

for p in net.collect_params().values():
    if p.grad_req != 'null':
        differentiable_params.append(p)

instep = 50
for e in range(epochs):
    loss_ = 0.0
    batch = 0
    for i, data in enumerate(train_dataloader):

        step_num += 1
        if step_num < num_warmup_steps:
            new_lr = lr * step_num / num_warmup_steps
        else:
            offset = (step_num - num_warmup_steps) * lr / \
                (num_train_steps - num_warmup_steps)
            new_lr = lr - offset
        trainer.set_learning_rate(new_lr)

        with mx.autograd.record():
            inputs, token_types, valid_length, start_label, end_label = data

            out = net(inputs.astype('float32').as_in_context(
                ctx), token_types.astype('float32').as_in_context(ctx), valid_length.astype('float32').as_in_context(ctx))

            ls = loss_function(out, [start_label.astype(
                'float32').as_in_context(ctx), end_label.astype('float32').as_in_context(ctx)]).mean()
        ls.backward()

        grads = [p.grad(ctx) for p in differentiable_params]
        gluon.utils.clip_global_norm(grads, 1)
        trainer.step(1)

        loss_ += ls.asscalar()
        batch += 1
        if (i+1) % instep == 0:
            print('Epoch: %d, Batch: %d, Loss: %.4f' % (e, i, loss_/(batch)))
            loss_ = 0.0
            batch = 0

_Result = collections.namedtuple(
    "_Result", ["qas_id", "start_logits", "end_logits"])

all_results = {}


for f in tqdm(dev_dataloader):
    input_ids = nd.reshape(nd.array(f.input_ids), (1, -1)
                           ).astype('float32').as_in_context(ctx)
    token_types = nd.reshape(nd.array(f.segment_ids),
                             (1, -1)).astype('float32').as_in_context(ctx)
    valid_length = nd.reshape(nd.array([f.valid_length]), (1,)).astype(
        'float32').as_in_context(ctx)
    out = net(input_ids, token_types, valid_length)
    output = nd.split(out, axis=0, num_outputs=2)

    all_results[f.qas_id] = _Result(f.qas_id, output[0].reshape(
        (-3, 0)).asnumpy(), output[1].reshape((-3, 0)).asnumpy())


all_predictions, all_nbest_json = predictions(
    dev_dataset=dev_dataset1, all_results=all_results)

f = open('predict.json', 'w', encoding='utf-8')
f.write(json.dumps(all_predictions))
f.close()
