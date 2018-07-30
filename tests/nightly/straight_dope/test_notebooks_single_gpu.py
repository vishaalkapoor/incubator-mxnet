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

#pylint: disable=no-member, too-many-locals, too-many-branches, no-self-use, broad-except, lost-exception, too-many-nested-blocks, too-few-public-methods, invalid-name, missing-docstring
"""
    This file tests that the notebooks requiring a single GPU run without
    warning or exception.
"""
import glob
import re
import os
import unittest
from straight_dope_test_utils import _test_notebook
from straight_dope_test_utils import _download_straight_dope_notebooks

NOTEBOOKS_WHITELIST = [
    'chapter01_crashcourse/preface',
    'chapter01_crashcourse/introduction',
    'chapter01_crashcourse/chapter-one-problem-set',
    'chapter02_supervised-learning/environment',
    'chapter03_deep-neural-networks/kaggle-gluon-kfold',
    'chapter07_distributed-learning/multiple-gpus-scratch',
    'chapter07_distributed-learning/multiple-gpus-gluon',
    'chapter07_distributed-learning/training-with-multiple-machines',
    'chapter12_time-series/intro-forecasting-gluon',
    'chapter12_time-series/intro-forecasting-2-gluon',
    'chapter13_unsupervised-learning/vae-gluon',
    'chapter18_variational-methods-and-uncertainty/bayes-by-backprop-rnn',
    'chapter17_deep-reinforcement-learning/DQN',
    'chapter17_deep-reinforcement-learning/DDQN',
    'chapter19_graph-neural-networks/Graph-Neural-Networks',
    'chapter16_tensor_methods/tensor_basics',
    'cheatsheets/kaggle-gluon-kfold'
]


class StraightDopeSingleGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        assert _download_straight_dope_notebooks()



    # TODO(vishaalk): Load params and Save params are deprecated warning.
    #def test_serialization(self):
    #    assert _test_notebook('chapter03_deep-neural-networks/serialization')

    # TODO(vishaalk): Load params and Save params are deprecated warning.
    #def test_deep_cnns_alexnet(self):
    #    assert _test_notebook('chapter04_convolutional-neural-networks/deep-cnns-alexnet')

    def test_rnns_gluon(self):
        assert _test_notebook('chapter05_recurrent-neural-networks/rnns-gluon')


    # TODO(vishaalk): RuntimeWarning: Overflow encountered in reduce.
    #def test_gd_sgd_scratch(self):
    #    assert _test_notebook('chapter06_optimization/gd-sgd-scratch')

    # TODO(vishaalk): RuntimeWarning: Overflow encountered in reduce.
    #def test_gd_sgd_gluon(self):
    #    assert _test_notebook('chapter06_optimization/gd-sgd-gluon')

    # TODO(vishaalk): Load params and Save params are deprecated warning.
    #def test_object_detection(self):
    #    assert _test_notebook('chapter08_computer-vision/object-detection')

    def test_fine_tuning(self):
        assert _test_notebook('chapter08_computer-vision/fine-tuning')

    def test_visual_question_answer(self):
        assert _test_notebook('chapter08_computer-vision/visual-question-answer')


    # TODO(vishaalk): Deferred initialization failed because shape cannot be inferred.
    #def test_intro_recommender_systems(self):
    #    assert _test_notebook('chapter11_recommender-systems/intro-recommender-systems')

     TODO(vishaalk): Investigate.
    def test_pixel2pixel(self):
        assert _test_notebook('chapter14_generative-adversarial-networks/pixel2pixel')

    # Chapter 18

    def test_bayes_by_backprop(self):
        assert _test_notebook('chapter18_variational-methods-and-uncertainty/bayes-by-backprop')

    def test_bayes_by_backprop_gluon(self):
        assert _test_notebook('chapter18_variational-methods-and-uncertainty/bayes-by-backprop-gluon')
