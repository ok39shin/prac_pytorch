# pytest 使ってみたら使えなかった
## pytestとは
- pythonモジュールの一つ
- pythonスクリプトの小さい単位でテストができる
- unittestとかもある
- `pytest -s $testdir`で実行
- $testdir内のtest\_\*.pyの`def test_*`関数という名前の条件に当てはまるものを実行する

## problem
- model.pyのテストをしようとtestディレクトリを作り，その中にmodel.pyの関数をimportして利用したtest\_model.pyを作成(`test/test_model.py`)
- `pytest -s test`を実行するとimportエラーmodelモジュールが存在しないと言われる
- インタラクティブモードでは`import model`は可能だった
- カレントディレクトリの`__init__.py`をrenameし，testディレクトリの中に`__init__.py`を作成(`touch test/__init__.py`)
- この状態で`pytest -s test`は実行できた
- しかし，カレントディレクトリでrenameした`__init__.py`の名前を戻し実行すると再びimportエラー <- 今ここ

## tips
- 小林さんは`__init__.py`がなくても動いてる
