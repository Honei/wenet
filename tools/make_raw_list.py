#!/usr/bin/env python3

# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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

import argparse
import json

if __name__ == '__main__':
    # segments参数使用--segments形式传入
    # wav_file, text_file和output_file是依次按照顺序传入进来
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--segments', default=None, help='segments file')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('text_file', help='text file')
    parser.add_argument('output_file', help='output list file')
    args = parser.parse_args()

    # 读取wav.scp内容，里面有音频的路径
    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            wav_table[arr[0]] = arr[1]

    if args.segments is not None:
        segments_table = {}
        with open(args.segments, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 4
                segments_table[arr[0]] = (arr[1], float(arr[2]), float(arr[3]))

    # 读取标注文本
    # 最后输出的格式是一个json，有三个字段
    # key: 音频的名称
    # wav: 音频的绝对路径
    # txt: 音频的标注文本
    with open(args.text_file, 'r', encoding='utf8') as fin, \
         open(args.output_file, 'w', encoding='utf8') as fout:
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]
            txt = arr[1] if len(arr) > 1 else ''
            if args.segments is None:
                assert key in wav_table
                wav = wav_table[key]
                line = dict(key=key, wav=wav, txt=txt)
            else:
                assert key in segments_table
                wav_key, start, end = segments_table[key]
                wav = wav_table[wav_key]
                line = dict(key=key, wav=wav, txt=txt, start=start, end=end)
            json_line = json.dumps(line, ensure_ascii=False)
            fout.write(json_line + '\n')
