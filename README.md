# music

### 概要
* Google発のMagentaをベースとして人工知能に音楽生成をする。
* 様々なRNNのモデル構造がある中、Magentaに新しいのモデル構造の取り入れ、音楽生成の質の向上を目指す。
  
### 気づき 
MagentaでMIDIファイルを食わせようとしたところ、複数音のものは食わせられない。これのモデルを作成する（12/20）
WAVEからの生成をおこなっていたが、モデルをかなり多きくしないといけないく、大きくしてしまうと、メモリー不足＆学習が終わらないため断念。また、フーリエ変換から食わせるのは、小さいデータがつぶれてしまうのでうまく行かない。

### 検証したいこと
* 転移学習による、学習時間削減（ジャンルごと）
* 一つの音楽のみを過学習させ、それに対してdrop outをかけて生成する（従来の逆のアプローチ）
* DQNを用いた音楽生成時に、今の状態でこの音（複数音）を鳴らした時の評価を求めるモデルを作成する

### 学内URL
* [PCの状態](http://hubble.ic.kanazawa-it.ac.jp/zabbix)(pascal2)

### Magentaについて
* [DQN](http://qiita.com/gdmiwasawa/items/362664b2e742281c1683)

### ディレクトリの説明
launcher：基本的にここの中にあるコードから走らせる
converter:データの変換を行うコード
magenta:magenta
midi:ダウンロードしたmidiの保存場所
model:DeepLearningのモデル
sample:modelのサンプルコード
test:使ってない
train_log:学習結果をおくディレクトリ
