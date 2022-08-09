# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from wenet.transducer.joint import TransducerJoint
from wenet.transducer.transducer import Transducer
from wenet.transducer.predictor import RNNPredictor
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import (TransformerDecoder,
                                       BiTransformerDecoder)
from wenet.transformer.encoder import (ConformerEncoder, TransformerEncoder)
from wenet.utils.cmvn import load_cmvn

def init_model(configs):
    """
        根据模型的配置参数初始化一个模型
        该模型只支持 ctc/attention 的结构
    """

    # 1. 加载cmvn，并将cmvn的计算过程转换为一个网络
    if configs['cmvn_file'] is not None:
        # load_cmvn 加载之后，是一个
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None
    
    # 2. 确定整个ASR的输入和输出维度
    #    输入维度目前是FBANK的梅尔滤波器数目
    #    输出维度是词典的数目
    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    # 3. 确定encoder和decoder网络 
    #    默认的encoder是conformer
    #    默认的decoder是bitransformer
    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    # 目前 encoder 只支持 conformer和transformer这两种，其他都不支持
    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    
    # 有decoder，使用decoder将encoder的输出转换为一个词典数目维度
    #  目前decoder的部分只支持transformer 和 bitransformer
    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])
    
    # 3. 最后的loss function 使用 ctc
    #    整个模型结构只有ctc这一个loss
    ctc = CTC(vocab_size, encoder.output_size())

    # Init joint CTC/Attention or Transducer model
    if 'predictor' in configs:
        predictor_type = configs.get('predictor', 'rnn')
        if predictor_type == 'rnn':
            predictor = RNNPredictor(vocab_size, **configs['predictor_conf'])
        else:
            raise NotImplementedError("only rnn type support now")
        configs['joint_conf']['enc_output_size'] = configs['encoder_conf'][
            'output_size']
        configs['joint_conf']['pred_output_size'] = configs['predictor_conf'][
            'output_size']
        joint = TransducerJoint(vocab_size, **configs['joint_conf'])
        model = Transducer(vocab_size=vocab_size,
                           blank=0,
                           predictor=predictor,
                           encoder=encoder,
                           attention_decoder=decoder,
                           joint=joint,
                           ctc=ctc,
                           **configs['model_conf'])
    else:
        # 默认情况下只有encoder和docoder，没有rnn-t这种方式
        # model_conf 中保留的是ctc和attention之间的关系
        model = ASRModel(vocab_size=vocab_size,
                         encoder=encoder,
                         decoder=decoder,
                         ctc=ctc,
                         **configs['model_conf'])
    return model
