# music

### 概要
* Google発のMagentaをベースとして人工知能に音楽生成をする。
* 様々なRNNのモデル構造がある中、Magentaに新しいのモデル構造の取り入れ、音楽生成の質の向上を目指す。

### 研究内容1
 すべてのジャンルの音楽を学ばせたあとに、そのモデルに一つ層を追加して特定のジャンルだけで再学習させる。
 この時に、学習対象とする層は追加した層だけにするのか、学習済みの層の学習率を下げるのか、全体同じ学習率にするのかも検証する。
 
### 研究内容2
 WAVEからの音楽生成を行う。　
 WAVEファイルにフーリエ変換を1s間隔で行い、サンプリングレートの1/2の周波数のみを用いて、学習させる。
 
### TODO（竹本）
* 学習させた最後の全結合層だけを取り除き、その他を特徴抽出器として扱うシステム設計
 * 学習済みパラメータ、モデルの保存、読み込み方法としてsaveメソッドを使用してみる
  

### TODO(oba)
issueで管理する。タイトルの最後にカッコをつけて期日を記載する

### TODO（論文としての項目）

  * magentaの各モデルの学習方法の違い（Basic RNN ,lookback RNN ,Attention RNN）
  * magentaのモデルを変えて、教育の改善をする（転移学習など）...細分化の余地あり
  * レイヤーによって学習率を変えれるかどうか
  * 音楽の細かなジャンル分け
  
### 気づき 
MagentaでMIDIファイルを食わせようとしたところ、複数音のものは食わせられない。これのモデルを作成する（12/20）
WAVEからの生成をおこなっていたが、モデルをかなり多きくしないといけないく、大きくしてしまうと、メモリー不足＆学習が終わらないため断念。また、フーリエ変換から食わせるのは、小さいデータがつぶれてしまうのでうまく行かない。
### 検証したいこと
* 転移学習による、学習時間削減（ジャンルごと）
* 一つの音楽のみを過学習させ、それに対してdrop outをかけて生成する（従来の逆のアプローチ）
* DQNを用いた音楽生成時に、今の状態でこの音を鳴らした時の評価を求めるモデルを作成する
##### 注意点
競合回避のため、新しくディレクトリを作成し、そこで作業する

### 学内URL
* [PCの状態](http://hubble.ic.kanazawa-it.ac.jp/zabbix)(pascal2)

### Magentaについて
* [DQN](http://qiita.com/gdmiwasawa/items/362664b2e742281c1683)
