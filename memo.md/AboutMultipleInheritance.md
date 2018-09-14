# About Multiple Inheritance for python
## Reason why I met this problem
- I wanted to make CNN-classifier model in pytorch.
- Already I had made CNN model(A) & NN-classifier model(B).
- I thought I want to inherit A & B to make CNN-classifier model, but it wouldn't working.

## environment
- python 3.6.5
- pytorch 0.4.1

## Problems
1. `class CNN2d_classifier(CNN2d, NN_classifier):`で多重継承したクラスを定義
2. 継承した親クラスの`__init__()`を実行
  2.1. `super().__init__(args)`で左から順に継承した親クラスの`__init__()`を探し，最初に見つけた`__init__()`を実行する(この場合CNN2dの`__init__()`を実行する)
  2.2. `super(CNN2d, self).__init__(args)`でCNN2dの次の親クラスから`__init__()`を探し，実行する(NN\_classifierの`__init__()`を実行してくれると思っていた)
3. うまくいかないことが判明
  3.1. どうやらCNN2d内の`super(CNN2d, self).__init__()`でエラー(nn.Moduleの`__init__()`を見るはず)
  3.2. CNN2dの次に見るクラスを確認`CNN2d_classifier.mro()`をプリント
  3.3. すると，CNN2dの次に見るクラスがnn.Moduleではなく，NN\_classifierだった
  3.4. 回避する方法を探すも発見できず
4. 親クラスのメソッドを実行する他の方法`SprCls.__init__(self, args)`を試す
  4.1. 同じエラーが出たが，出た場所が変わった
  4.2. 他のsuper()を使っている部分も同じく変更したところ，うまく動作した

## 懸念
- 今回は問題に対処はできたが，superを使わないことによる弊害とかがあるんじゃないかと思っている．(だったらsuper使う意味ないし)
- まだまだpythonは奥が深い
