# Simple TD-MPCs (日本語版)

[![TD-MPC](http://img.shields.io/badge/tdmpc-arxiv.2203.04955-B31B1B.svg)](https://arxiv.org/abs/2203.04955)
[![TD-MPC2](http://img.shields.io/badge/tdmpc2-arxiv.2310.16828-B31B1B.svg)](https://arxiv.org/abs/2310.16828)


JAX と Flax による TD-MPC シリーズの非公式実装リポジトリです。

TD-MPC2の公式リポジトリが複数タスクの学習に対応しているのに対し、本リポジトリはシンプルさに重点を置き、単一タスクの学習に特化したコードを提供しています。

また、CleanRLに触発され、アルゴリズムの処理を複数のファイルに分散させるのではなく、一つのファイル内でアルゴリズムの流れを追いやすいように設計し、可読性の向上を図っています。

## 実行結果
DM-Control環境でのプログラムのテスト結果は以下の通りです：

* TD-MPC

<アニメーション>

* TD-MPC2

<アニメーション 2>

詳細なログは以下のURLから確認できます：

<Wandb の URL>

## 環境設定

1. Dockerを使用して実行環境をセットアップします。

    1.1 Docker のビルド
        
    ```
    docker build -t simple_tdmpc .
    ```

    1.2 Docker の環境に入る

    ```
    docker run \
        --gpus all \
        -it \
        --rm \
        -w $HOME/work \
        -v $(pwd):$HOME/work \
        simple_tdmpc:latest bash
    ```


2. Poetryを利用して依存ライブラリをインストールします。
    ```
    poetry install
    ```

## 実行方法

1. TD-MPC の実行

    ```
    poetry run python src/tdmpc.py 
    ```
2. TD-MPC2 の実行

    ```
    poetry run python src/tdmpc2.py 
    ```

## オプション

1. `--capture_video` オプションを付与すると、学習プロセス中のビデオを `/videos` フォルダに保存できます。

    ```
    poetry run python src/tdmpc2.py --capture_video
    ```

2. `--track` オプションを付与すると、wandb を介して実験ログを記録できます。

    ```
    poetry run wandb login
    poetry run python src/tdmpc2.py --track --capture_video
    ```
    > 二行目で `transport faild error` が出る場合は、エラー文に出力される ```git config``` コマンドを実行して下さい。

3. `--task` オプションで学習するタスクを切り替えることができます。

    ```
    poetry run python src/tdmpc2.py --task 'dm_control/quadruped-run-v0' --total_timesteps 1000000
    ```


## 参照リポジトリ

- [TD-MPC](https://github.com/nicklashansen/tdmpc)
- [TD-MPC2](https://github.com/nicklashansen/tdmpc2)
- [CleanRL (Clean Implementation of RL Algorithms)](https://github.com/vwxyzjn/cleanrl)