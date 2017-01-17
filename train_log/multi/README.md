# 概要
`midi/multi_train_midi`のディレクトリ内にある曲を過学習させた時のlogと、そのモデルとモデルを使った音楽生成のデータ保管場所。
音楽の生成は`launcher/multi_midi_generate_launcher.py`を用いて行った。
生成方法は、学習させたデータ(`midi/multi_train_midi`)の半分の時間を入力として与え、残りの半分を自分で生成させるということをした。
tensorboardで表示するログは容量の関係上、圧縮してある。log見たいときは`tar zxvf events.out.tfevents.1484150505.pascal1.tgz`を実行すること。

# generated
generatedには実際に生成した音楽が入っている

* v1: 閾値を決めてその値より大きいものは全部出力する(閾値0.5で生成している)
* v2: v1の状態だと、あるフレーズから次のフレーズに移る場合、複数のフレーズの候補があった時、複数のフレーズすべての音をだしてしまい、うまく生成されていないと考えた。そこで、閾値以上の音のすべての組み合わせから最適な音の組み合わせを見つけてきて生成を行っている。

# Config
```py
class Config:
    batch_size = 1
    sequence_length = 1
    input_size = 128
    label_size = 128
    cell_size_list = [128, 128]
    keep_prob = 1.0
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3
```