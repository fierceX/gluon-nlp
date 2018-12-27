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
"""BERT models."""

__all__ = ['BERTSquad', 'BERTloss']

from mxnet.gluon import Block
from mxnet.gluon import nn, loss
from mxnet.gluon.loss import Loss


class BERTSquad(Block):
    def __init__(self, bert, num_classes=2, prefix=None, params=None):
        super(BERTSquad, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.Dense = nn.Dense(units=num_classes, flatten=False)

    def forward(self, inputs, token_types, valid_length=None):
        bert_output = self.bert(inputs, token_types, valid_length)
        output = self.Dense(bert_output)
        output = nd.transpose(output, (2, 0, 1))
        return output


class BERTloss(Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BERTloss, self).__init__(weight=None, batch_axis=0, **kwargs)
        self.loss = loss.SoftmaxCELoss()

    def hybrid_forward(self, F, pred, label):
        pred = F.split(pred, axis=0, num_outputs=2)
        start_pred = pred[0].reshape((-3, 0))
        start_label = label[0]
        end_pred = pred[1].reshape((-3, 0))
        end_label = label[1]
        return (self.loss(start_pred, start_label)+self.loss(end_pred, end_label))/2
