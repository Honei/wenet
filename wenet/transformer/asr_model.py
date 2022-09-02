# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from torch.nn.utils.rnn import pad_sequence

from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)


# ASRModel 的详细解释参见: https://zhuanlan.zhihu.com/p/381095506?utm_source=wechat_session&utm_medium=social&utm_oi=34522477363200
class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model
        在 ASRModel中，包装了ASR 的处理过程
        ASRModel是一个ctc/Attention的混合模型
    """
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):
        """
            vocab_size: 表示词典
            encoder: encoder+ctc框架中encoder的部分
            decoder: 使用attention解码的部分
            ctc: ctc 损失函数
            ctc_weight: ctc/attention框架中，ctc损失函数的权重
            ignore_id: 不同音频长度进行补齐时，补齐的符号
            reverse_weight: 在decoder中，如果有反向解码，计算decoder的loss时，方向解码时loss的权重
            lsm_weight: 在decoder中，使用label smoothing技术时，label smothing的值
            length_normalized_loss: 对decoder最后的loss进行归一化时，使用batch_size归一化还是使用有效字符的长度进行归一化
                                    设置为True时，表示使用有效字符的长度进行归一化
        """
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        # 在ASR中sos和eos使用相同的符号
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight    # 反向解码的时候，反向loss权重
        
        # 确定encoder， decoder， loss function和 label smoothing loss
        # 训练过程中需要的网络部分都有了
        # 目前encoder选择ConformerEncoder
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc

        # attention分支中，使用LabelSmoothing技术
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        #    在正常训练的过程中，可以不需要 encoder+decoder 框架中的decoder
        #    speech的维度信息是 (batch, times, feature_dim)
        #    wenet使用Fbank，维度是80
        #    那么每次计算时维度为(batch, times, 80)
        #    speech_lengths 表示每个音频的长度，这里长度是语音帧的数目
        #    目前 encoder 选择 ConformerEncoder
        #    encoder_mask 的维度是(batch, 1, time)
        #    这里的time是经过下采样之后的值，例如经过1/4的下采样，那么time=T/4，T是原始的长度
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        
        # encoder_out_lens 是每个音频语音序列的长度，维度是(batch)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            # loss_att 是loss function
            # acc_att是符号的正确率
            loss_att, acc_att = self._calc_att_loss(encoder_out, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            # 两个方式联合起来，得到最后的attention结果
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return {"loss": loss, "loss_att": loss_att, "loss_ctc": loss_ctc}

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
            计算ctc的损失函数值
            encoder_out: (batch, time, d_model) time是经过下采样之后序列的长度，d_model是attention之后的维度
            encoder_masks: (batch, 1, time) 标记了每个序列中有效位置
        """

        # ys_in_pad是在每个序列的前面添加sos, 尾部用eos填充
        # ys_in_pad是attention的输入符号
        # ys_out_pad是在每个序列的后面添加eos, 尾部用self.ignore_id填充
        # ys_out_pad是用来用来计算loss，所有的符号以eos结尾
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                               self.ignore_id)

        # ys_in_lens的长度需要添加1，每个序列都添加了一个sos符号
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        # r_ys_pad 是对ys_pad进行翻转
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        # r_ys_in_pad是在每个翻转序列的前面添加sos,尾部用eos填充
        # r_ys_out_pad是在每个翻转序列的后面添加eos，尾部用self.ignore_id填充
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)

        # ys_in_pad和r_ys_in_pad是前向和反向序列
        #  使用ys_in_pad和r_ys_in_pad双向列表计算decoder
        #  这种情况只能在非流失的情况下效果很好
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)
        # 2. Compute attention loss
        #    计算attention解码的结果loss值
        loss_att = self.criterion_att(decoder_out, ys_out_pad)

        # 在Transformer中，r_loss_att为0
        # 这样可以支持流式解码
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        
        # 计算正向loss和反向loss
        # 得到最终的loss
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        
        # 统计符号的正确率
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        # 由于流式解码和非流式解码过程不同
        # 因此这里将流式和非流失分开，不是直接调用forward
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            # 非流失解码，直接输出encoder_out
            # 在非流式解码中，speech_lengths的长度就是音频自身的长度
            # 因此在解码的时候，输出的encoder_mask全部都是True
            # 默认情况下decoding_chunk_size=-1,表示使用所有的历史数据
            #         num_decoding_left_chunks
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def attention_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        #    第一步是使用encoder计算中间的特征向量
        #    encoder_out 的维度是(batch, times, dim)
        #    encoder_out 相当于降采样之后的(h_1, h_2, ..., h_{U})
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)        # 解码的时候，maxlen是序列的长度
        encoder_dim = encoder_out.size(2)   # 中间特征维度

        #    扩展beam_size个相同的encoder_out结果
        #    由于要输出beam_size个结果，因此需要将encoder_out重复beam_size次
        #    这样之前解码出来的每个句子hyp都有相同的当前语音帧的结果
        running_size = batch_size * beam_size   # running_size  表示扩展之后句子的数目
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
            running_size, maxlen, encoder_dim)  # (B*N, maxlen, encoder_dim)
        encoder_mask = encoder_mask.unsqueeze(1).repeat(
            1, beam_size, 1, 1).view(running_size, 1,
                                     maxlen)  # (B*N, 1, max_len)

        # 开始的句子里面都是填充的 <sos> 符号
        # hyps 是所有batch的候选句子
        hyps = torch.ones([running_size, 1], dtype=torch.long,
                          device=device).fill_(self.sos)  # (B*N, 1)
        
        # 第一个符号的概率为0.0，概率是对数概率，
        # 默认第0个符号的概率是1.0，即解码图的<sos>符号概率是1.0
        # 转换为对数概率是0.0，
        # 其他的概率都是0.0，那么对数概率是-inf
        # scores 表示历史记录beam个句子的分数
        # 解码开始的时候，第一个符号是<sos>，在第一次解码的时候，都是从<sos> 开始解码
        # 只能将 beam_size 个候选位置了，任意一个设置为概率0.0，否则会导致beam search找出10个top1的句子
        # 每个batch都要进行处理依次
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                              dtype=torch.float)
        scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(
            device)  # (B*N, 1)

        # end_flag 用来记录解码出 <eos> 的句子数目
        #          开始的时候，没有一个句子解码到了 <eos> 符号
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)
        # cache 中记录的是历史解码单元的内容
        # cache 是一个列表，列表中每个元素是decoder中每一层的历史信息
        # 这样使用self.decoder.forward_one_step 进行解码时，利用了所有已经解码出来的数据
        # 这也会导致，随着音频的增长，解码一帧数据需要的时间会变的很长
        cache: Optional[List[torch.Tensor]] = None
        
        # 2. Decoder forward step by step
        #    依次进行解码
        #    beam search 解码的时候，一共有 beam 个候选句子
        #    于此同时，每个候选句子在每一步的时候，都有 beam 个候选
        for i in range(1, maxlen + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            
            # 2.1 Forward decoder step
            #     subsequent_mask(i) 返回自回归方式下，每个序列和之前序列的依赖关系
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(
                running_size, 1, 1).to(device)  # (B*N, i, i)

            # logp: (B*N, vocab)
            # 在第一个符号<sos>输入进去之后，所有的解码结果都是相同的
            # 使用hyps和encoder_out进行打分，得到hyps和每个encoder_out之间的分数
            # hyps 是 batch_size * beam 个候选句子
            # cache 中记录的是历史解码单元的内容
            # cache 是一个列表，列表中每个元素是decoder中每一层的历史信息
            # 这样使用self.decoder.forward_one_step 进行解码时，利用了所有已经解码出来的数据
            # 这也会导致，随着音频的增长，解码一帧数据需要的时间会变的很长
            logp, cache = self.decoder.forward_one_step(
                encoder_out, encoder_mask, hyps, hyps_mask, cache)
            
            # 2.2 First beam prune: select topk best prob at current time
            #     开始解码剪枝，只选取概率最高的beam_size个结果
            #     经过变换之后，top_k_logp 的维度是(B*N, N)
            #     达到每个候选的句子都有的beam个候选
            #     然后这beam候选句子会从 beam * beam 个候选中重新挑选出分数最高的 beam 个候选
            #     作为新的 beam 个候选
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)

            # 2.3 Second beam prune: select topk score with history
            #     每个句子的历史分数和当前语音帧分数top_k_logp相加
            #     得到的是添加新的符号之后，新的句子的分数
            #     对数概率，原本是概率相乘，这时就是概率相加
            
            # scores 是之前每个句子历史分数，每个句子输入到attention之后，得到各自的beam_size个候选
            # scores + top_k_logp 这样，beam_size 每个句子都重新有 beam_size 个候选，
            # 需要重新在 beam_size*beam_size 中挑选出新的 beam_size 个候选
            scores = scores + top_k_logp  # (B*N, N), broadcast add

            #  展开之后，每个句子是有N*N个新的候选
            #  每个句子下N*N个新的候选中重新挑选出N个候选
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            
            # 每个句子从 beam_size * beam_size 里面挑选beam_size个分数最高的符号概率
            # offset_k_index是在N*N中的候选的序号
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)

            # 每个句子中选取最大的分数作为新的beam个句子
            scores = scores.view(-1, 1)  # (B*N, 1)

            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            # base_k_index 是每个候选句子 batch 的索引
            # batch 中每个句子有 beam_size * beam_size 个新的候选
            # 每个句子是在 beam_size * beam_size 个候选中挑选出 beam_size 个候选
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            
            # 因此 base_k_index * beam_size * beam_size 
            # 是每个句子新挑选出句子时，在 batch * beam_size * beam_size 所有句子中的基础偏移
            # 例如：batch 中第一个候选句子基础偏移是0
            #      batch 中第二个句子基础偏移是 1 * beam_size * beam_size, 前 beam_size * beam_size 是属于第一个句子的
            base_k_index = base_k_index * beam_size * beam_size

            # best_k_index + offset_k_index 这样就得到了每个句子绝对位置的偏移
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            #     解码得到最后的句子
            #     在 top_k_index中挑选出序号是 besk_k_index 的那些位置内容
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            
            # best_hyps_index得到的是原始beam_size个候选句子的序号
            best_hyps_index = best_k_index // beam_size
            # 通过 best_hyps_index 获取上次的候选句子内容last_best_k_hyps
            # 将 last_best_k_hyps 和当前选出来的解码单元 best_k_pred 合并在一起就是新的beam_size个候选句子
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            # 更新end_flag，判断解码出来的句子中，哪一个句子解码得到了eos
            end_flag = torch.eq(hyps[:, -1], self.eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]
        return best_hyps, best_scores

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ Apply CTC greedy search
            CTC的贪婪解码，是最简单的解码方式
        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        # speech的维度是:(batch, times, feat_dim)
        # speech_lengths的维度是:(batch), 只有一个常量，常量的值就是batch的数目
        # 在解码阶段，batch_size 通常是1
        batch_size = speech.shape[0]

        # Let's assume B = batch_size
        # 1. 经过encoder推理之后，得到一个新的特征数据
        #    维度信息是(batch, times, encoder_dim)
        #    encoder_dim是encoder内部attention的维度
        #    在理论上，这里的输出结果是(h_{1}, h_{2}, ..., h_{U})
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)

        # 2. 进行贪婪解码
        #   在解码阶段，encoder_mask全部都是True
        #   encoder_mask的维度是：(1, 1, times)
        #   encoder_out_lens 和 maxlen 是相同的
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
  
        # 3. 根据encoder内部解码的信息，计算ctc的概率
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        
        # 4. 直接选取每个时间序列上的概率最大值，作为每个时间的实际输出结果
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        # topk_index 是每个语音帧上概率最大的序号,也是最后的解码结果
        # topk_index 的维度是(B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        # 将所有补零的地方设置为eos, 在解码的时候，没有位置是eos
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        
        # 5. 解码结果通常只有1个
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores

    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation
            ctc 的beam search 解码方式
        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # 在解码的时候，只有一个音频同时解码
        # 这个接口在 prefix_beam_search 和 attention_rescoring 里面都复用
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        #    通常是非流式解码，直接输出encoder_out
        #    encoder_out的维度信息是(batch, times, attn_dim)
        #    encoder_out 解码出来的结果，在理论上是(h_1, h_2, ..., h_{U})
        #    U 是经过下采样之后的音频序列长度
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)

        # self.ctc.log_softmax的部分是原始ctc中的内容
        # 在计算log_softmax的时候，ctc里面没有使用dropout
        # 因为使用dropout，经过softmax之后，并不影响最后的结果
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        # 在解码的规整字符串中，将空字符串的概率初始化为1，非空串的概率初始化为0.0
        # 转换为对数概率之后，以空格结尾的对数概率是0.0， 非空格的对数概率为-inf
        # cur_hyps 记录了当前解码的句子，初始化是一个空句子
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        #    进行ctc解码, ctc解码是动态规划的方法
        for t in range(0, maxlen):
            # 取出当前语音帧的概率，这一帧数据包括所有符号的概率
            logp = ctc_probs[t]  # (vocab_size,)
            
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            #     依次遍历每个 topk 的解码单元
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                # s 是这一帧概率最高的beam_size个符号对应的id，
                s = s.item() # 将torch中的tensor转换为int值
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        # 当 t+1 输出是blank是，产生输出符号 *s，更新以 blank 结尾的概率
                        n_pb, n_pnb = next_hyps[prefix]

                        # log_add([pb + ps, pnb + ps]) 是 cur_hyps 中以 prefx 为前缀的序列中，以空格结尾的所有解码路径概率和
                        # n_pb 当前语音帧已经解码到的 prefix 为前缀的所有解码路径概率之和
                        # 因此这里要将所有 prefix 为前缀的路径概率都进行相加
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])

                        # 对于n_pnb，那么如果之前有新的解码序列得到n_pnb，那么当前n_pnb，是用之前的n_pnb
                        # 否则n_pnb就是-inf，即对数概率是n_pnb, 对应的概率是0
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        # 当 t+1 输出是s是，可以产生输出符号 *s，更新以 s 结尾的概率
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        
                        # Update *s-s -> *ss, - is for blank
                        # 也可以产生出输出符号 *ss， 更新以 s 结尾的 *ss 的概率
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        # Update new symbol
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            # 根据每个解码序列进行求和，重新剪枝，只保留新的beam_size个解码符号
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            
            # 更新解码序列
            cur_hyps = next_hyps[:beam_size]

        # 将解码序列和概率，转换元组的形式
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out

    def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]:
        """ Apply CTC prefix beam search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        # 调用 ctc 的 prefix beam search 解码算法
        # 最后返回一个分数最高的解码序列
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming)
        return hyps[0]

    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
   
        # For attention rescoring we only support batch_size=1
        # 对于 attention rescoring 解码算法，目前只支持 batch size=1的解码方法
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        # encoder_dim 是内部attention的维度
        # encoder_out 相当于 (h_1, h_2, ..., h_{U})
        # U 是降采样之后音频序列的长度
        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)

        assert len(hyps) == beam_size
        # 对 ctc 解码出来的 beam size 个序列进行补齐
        # 这样在使用recoring的时候，每个句子是等长的
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)

        # 获取 pad 的信息，将所有的 self.ignore_id 替换为 self.eos
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining，每个序列的开始位置都添加了 <sos> 符号
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        
        # used for right to left decoder
        # 使用解码出来的句子，和原始的内部特征进行attention计算
        # 得到新的decoder_out解码
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        # 经过 decoder 运算之后，得到的是decoder的输出(s_1, s_2, ..., s_{U})
        # 使用 decoder 解码之后，可以直接解码得到字单元的概率
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        
        # Only use decoder score for rescoring
        # 每个句子和自身的特征进行attention打分，得到新的分数
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0

            # 得到 decoder 网络每个单元的对数概率
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                # 更新 attention 的分数
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            # 总的分数是 attention 的分数+CTC的分数
            score += hyp[1] * ctc_weight
            
            # 更新最优分数的序号
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0], best_score

    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output

        Returns:
            torch.Tensor: activation before ctc

        """
        return self.ctc.log_softmax(xs)

    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if hasattr(self.decoder, 'right_decoder'):
            return True
        else:
            return False

    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)

        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        #   >>> r_hyps
        #   >>> tensor([[ 1,  2,  3],
        #   >>>         [ 9,  8,  4],
        #   >>>         [ 2, -1, -1]])
        #   >>> r_hyps_lens
        #   >>> tensor([3, 3, 1])

        # NOTE(Mddct): `pad_sequence` is not supported by ONNX, it is used
        #   in `reverse_pad_list` thus we have to refine the below code.
        #   Issue: https://github.com/wenet-e2e/wenet/issues/1113
        # Equal to:
        #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        #   >>> seq_mask
        #   >>> tensor([[ True,  True,  True],
        #   >>>         [ True,  True,  True],
        #   >>>         [ True, False, False]])
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        #   >>> index
        #   >>> tensor([[ 2,  1,  0],
        #   >>>         [ 2,  1,  0],
        #   >>>         [ 0, -1, -2]])
        index = index * seq_mask
        #   >>> index
        #   >>> tensor([[2, 1, 0],
        #   >>>         [2, 1, 0],
        #   >>>         [0, 0, 0]])
        r_hyps = torch.gather(r_hyps, 1, index)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, 2, 2]])
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, eos, eos]])
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        #   >>> r_hyps
        #   >>> tensor([[sos, 3, 2, 1],
        #   >>>         [sos, 4, 8, 9],
        #   >>>         [sos, 2, eos, eos]])

        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
            reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

        # right to left decoder may be not used during decoding process,
        # which depends on reverse_weight param.
        # r_dccoder_out will be 0.0, if reverse_weight is 0.0
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        return decoder_out, r_decoder_out
