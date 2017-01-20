# issue
* sequence_length
* cell_size_list
* optimizer_function(Adam or GD)
の３っつのパラメータの最適な値を求める。
epochは500で行う

# 1

```py
sequence_length = 40
cell_size_list = [128]
optimizer_function = tf.train.AdamOptimizer(0.1)
```
間違えてえepoch1000のままで行ってしまった。
使い道があるかもしれないので取っておく。

# 2

```py
sequence_length = 40
cell_size_list = [128]
optimizer_function = tf.train.AdamOptimizer(0.1)
```

# 3

```py
sequence_length = 40
cell_size_list = [128, 128]
optimizer_function = tf.train.AdamOptimizer(0.1)
```

# 4
```py
sequence_length = 40
cell_size_list = [128]
optimizer_function = tf.train.GradientDescentOptimizer(0.1)
```

# 5
```py
sequence_length = 40
cell_size_list = [128, 128]
optimizer_function = tf.train.GradientDescentOptimizer(0.1)
```

#6
```py
sequence_length = 100
cell_size_list = [128]
optimizer_function = tf.train.GradientDescentOptimizer(0.1)
```

#7
```py
sequence_length = 100
cell_size_list = [128, 128]
optimizer_function = tf.train.GradientDescentOptimizer(0.1)
```


#8
一番うまく行っているように見える
```py
sequence_length = 20
cell_size_list = [128]
optimizer_function = tf.train.GradientDescentOptimizer(0.1)
```
学習時間:5m48s
33.4K step
loss:0.08

#9
```py
sequence_length = 20
cell_size_list = [128, 128]
optimizer_function = tf.train.GradientDescentOptimizer(0.1)
```
学習時間:9m7s
33.4K step
loss:0.12

#10
8がうまく行っているみたいなので、Adamでもやってみる
```py
sequence_length = 20
cell_size_list = [128]
optimizer_function = tf.train.AdamOptimizer(0.1)
```


#11
8がうまく行ったので、sequence_lengthをさらに小さくしてみる
```py
sequence_length = 10
cell_size_list = [128]
optimizer_function = tf.train.GradientDescentOptimizer(0.1)
```
経過時間:7m27s
66.40K step
loss:0.07

#12
11がうまく行ったので、さらに小さく！
```py
sequence_length = 5
cell_size_list = [128]
optimizer_function = tf.train.GradientDescentOptimizer(0.1)
```
学習時間:10m38s
132.9K step
lsoo:0.06

#13
```py
sequence_length = 5
cell_size_list = [128, 128]
optimizer_function = tf.train.GradientDescentOptimizer(0.1)
```
15m56s
132.9K step
loss:0.08

# 考察
sequence_lengthが大きいと一回の学習に時間がかかり、同じepochでも学習の回数が少なくなり、あまり効率が良くなかった。
逆にsequence_lengthを短くすることで時間あたりの学習時間を多くできるので効率よく学習できた。
sequence_lengthが大きいときそれに応じてlearning_rateもあげてもいいのかなと思った。
新たな学習方法で、sequence_lengthを徐々に大きくしていく学習方法もあるのかなと思った。(データセット作るのが大変)

Adamについては、時々lossが大きくなったのでこれについては疑問が残った。
Adamの方法について知る必要があると思った。

