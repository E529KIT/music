# 概要
「ザナルカンドにて」の曲を過学習させた時のlogと、そのモデルとモデルを使った音楽生成のデータ保管場所。
short_trainは最初の9小説のみを学習させた時のデータ。
full_trainは最初から最後まで学習させた時のデータ。
full_trainは学習時間短縮のためにshort_trainの学習済みモデルを使用している

# ディレクトリ構造
short_train, full_train以下のディレクトリについて説明する。
tensorboardを使って学習の経過とかみたい場合には、"...../sort_train"の位置で"tensorboard --logdir=./"と入力することで見ることができる。

## generated
学習済みモデルを使用して生成した音楽が入っている。
ファイル名は"dropoutN_lenM.mid"の形をしていている。
Nはドロップアウトのレートを示しており、0.Nのドロップアウトラッパーをかけているということを示している。
Mは生成した音楽の時間を示しており、学習時に使用した音楽のながさのM倍の音楽を生成していることを示している。
また、Mが1の時は最初の一つの音のみから作成しており、Mが２の時は、元の音楽全てを入力として入れている。

## data
学習済みのモデルのデータが入ってる。
Saverを使ってモデルをrestoreしたい時には"data/model"を指定することで学習済みモデルを使用することができる。

# ハイパーパラメータ
```py
    batch_size = 1
    sequence_length = 20
    input_size = 128
    label_size = 128
    cell_size_list = [128, 128]
    keep_prob = 0.9
    optimizer_function = tf.train.GradientDescentOptimizer(0.1)
    clip_norm = 3
```