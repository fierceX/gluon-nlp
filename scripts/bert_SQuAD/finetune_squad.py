"""
SQuAD with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
SQuAD, with Gluon NLP Toolkit.

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import collections
import json
import logging
import random
import time

import gluonnlp as nlp
import mxnet as mx
import numpy as np
from mxnet import gluon, nd

from bert import BERTloss, BERTSquad
from dataset import SQData, SQuADTransform
from evaluate import evaluate, predictions

np.random.seed(0)
random.seed(0)
mx.random.seed(2)
logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='BERT QA example.'
                                             'We fine-tune the BERT model on SQuAD 1.1')

parser.add_argument('--epochs', type=int, default=2, help='number of epochs')

parser.add_argument('--batch_size', type=int, default=12,
                    help='Batch size. Number of examples per gpu in a minibatch')

parser.add_argument('--test_batch_size', type=int,
                    default=8, help='Test batch size')

parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimization algorithm')
parser.add_argument('--lr', type=float, default=3e-5,
                    help='Initial learning rate')

parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')

parser.add_argument('--log_interval', type=int,
                    default=50, help='report interval')

parser.add_argument('--max_seq_length', type=int, default=384,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                    'Sequences longer than this will be truncated, and sequences shorter '
                    'than this will be padded.')

parser.add_argument('--doc_stride', type=int, default=128,
                    help='When splitting up a long document into chunks, how much stride to '
                    'take between chunks.')

parser.add_argument('--max_query_length', type=int, default=64,
                    help='The maximum number of tokens for the question. Questions longer than '
                    'this will be truncated to this length.')

parser.add_argument('--n_best_size', type=int, default=20,
                    help='The total number of n-best predictions to generate in the '
                    'nbest_predictions.json output file.')

parser.add_argument('--max_answer_length', type=int, default=30,
                    help='The maximum length of an answer that can be generated. This is needed '
                    'because the start and end predictions are not conditioned on one another.')

parser.add_argument('--version_2', type=bool, default=False,
                    help='If true, the SQuAD examples contain some that do not have an answer.')

parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                    help='If null_score - best_non_null is greater than the threshold predict null.')

parser.add_argument('--gpu', action='store_true',
                    help='whether to use gpu for finetuning')


args = parser.parse_args()
logging.info(args)

epochs = args.epochs
batch_size = args.batch_size
test_batch_size = args.test_batch_size
lr = args.lr
ctx = mx.cpu() if not args.gpu else mx.gpu()
optimizer = args.optimizer
log_interval = args.log_interval
warmup_ratio = args.warmup_ratio

dataset_name = 'book_corpus_wiki_en_uncased'
version_2 = args.version_2
max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length
n_best_size = args.n_best_size
max_answer_length = args.max_answer_length
null_score_diff_threshold = args.null_score_diff_threshold


if max_seq_length <= max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (max_seq_length, max_query_length))

bert, vocab = nlp.model.bert_12_768_12(dataset_name=dataset_name,
                                       pretrained=True, ctx=ctx, use_pooler=False,
                                       use_decoder=False, use_classifier=False,)

berttoken = nlp.data.BERTTokenizer(vocab=vocab)

logging.info('Loader train data...')
train_data = SQData('./squad1.1/train-v1.1.json', version_2=version_2)
train_data = train_data.transform(
    SQuADTransform(berttoken, max_seq_length=max_seq_length, doc_stride=doc_stride,
                   max_query_length=max_query_length, is_training=True))
train_dataloader = mx.gluon.data.DataLoader(
    train_data, batch_size=batch_size, num_workers=4, shuffle=True)

net = BERTSquad(bert=bert)
net.Dense.initialize(init=mx.init.Normal(0.02), ctx=ctx)
net.hybridize(static_alloc=True)

loss_function = BERTloss()
loss_function.hybridize(static_alloc=True)


def Train():

    logging.info('Start training')

    trainer = gluon.Trainer(net.collect_params(), optimizer, {
                            'learning_rate': lr, 'epsilon': 1e-9})

    num_train_examples = len(train_data)
    num_train_steps = int(num_train_examples / batch_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    differentiable_params = []

    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    for p in net.collect_params().values():
        if p.grad_req != 'null':
            differentiable_params.append(p)

    for epoch_id in range(epochs):
        step_loss = 0.0
        tic = time.time()
        for batch_id, data in enumerate(train_dataloader):

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

            step_loss += ls.asscalar()

            if (batch_id+1) % log_interval == 0:
                toc = time.time()
                logging.info('Epoch: %d, Batch: %d/%d, Loss=%.4f, lr=%.7f Time cost=%.1f' %
                             (epoch_id, batch_id, len(train_dataloader),
                              step_loss/(log_interval), trainer.learning_rate, toc-tic))
                tic = time.time()
                step_loss = 0.0


def Evaluate():
    logging.info('Loader dev data...')
    dev_data = SQData('./squad1.1/dev-v1.1.json',
                      version_2=version_2, is_training=False)

    dev_dataset = dev_data.transform(SQuADTransform(
        berttoken, max_seq_length=max_seq_length, doc_stride=doc_stride,
        max_query_length=max_query_length, is_training=False)._transform)

    dev_data = dev_data.transform(SQuADTransform(
        berttoken, max_seq_length=max_seq_length, doc_stride=doc_stride,
        max_query_length=max_query_length, is_training=False))

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data, batch_size=batch_size, num_workers=4, shuffle=False)

    logging.info('Start predict')
    for data in tqdm(dev_dataloader):
        inputs, token_types, valid_length, _, _ = data

        out = net(inputs.astype('float32').as_in_context(
            ctx), token_types.astype('float32').as_in_context(ctx), valid_length.astype('float32').as_in_context(ctx))

        output = nd.split(out, axis=0, num_outputs=2)
        start_logits.extend(output[0].reshape((-3, 0)).asnumpy())
        end_logits.extend(output[1].reshape((-3, 0)).asnumpy())
    all_results = (start_logits, end_logits)

    all_predictions, all_nbest_json = predictions(
        dev_dataset=dev_dataset, all_results=all_results, max_answer_length=max_answer_length, tokenizer=nlp.data.BasicTokenizer(lower_case=True))

    f = open('predict.json', 'w', encoding='utf-8')
    f.write(json.dumps(all_predictions))
    f.close()

    logging.info(evaluate('./squad1.1/dev-v1.1.json', all_predictions))


if __name__ == '__main__':
    Train()
    net.save_parameters('./net')
    Evaluate()
