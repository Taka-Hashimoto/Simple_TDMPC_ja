# Simple TD-MPCs (日本語版)

[![TD-MPC](http://img.shields.io/badge/tdmpc-arxiv.2203.04955-B31B1B.svg)](https://arxiv.org/abs/2203.04955) [![TD-MPC2](http://img.shields.io/badge/tdmpc2-arxiv.2310.16828-B31B1B.svg)](https://arxiv.org/abs/2310.16828)

JAX と Flax による TD-MPC シリーズの非公式実装リポジトリです。

TD-MPC2の公式リポジトリが複数タスクの学習に対応しているのに対し、本リポジトリはシンプルさに重点を置き、単一タスクの学習にのみ対応しています。

CleanRLに触発されて本実装では、アルゴリズムの処理を複数のファイルに分散させるのではなく、一つのファイル内でアルゴリズムの流れを追いやすいように設計し、可読性の向上を図っています。

## 実行結果
DM-Control環境での学習テスト結果を以下に示します。

### 学習曲線

<div style="text-align: center">
<img width="80%" alt="tdmpc_vs_tdmpc2.png (157.6 kB)" src="https://img.esa.io/uploads/production/attachments/21189/2024/04/01/160121/706d85d0-37bc-47b6-8ea1-ee979f3c518f.png">
</div>

### アニメーション

<div style="text-align: center">
<figure>
<video controls width="19%" alt="tdmpc_cheetah.mp4 (405.3 kB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/34ef0c7a-8928-4e27-a804-d3827872c6c6.mp4" autoplay muted></video>
<video controls width="19%" alt="videos_9_80fc9aa78938edb0e05c.mp4 (1.4 MB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/51bc9a3b-6f16-48d5-a9aa-2401d04101e2.mp4" autoplay muted></video>
<video controls width="19%" alt="videos_49_fdaeea142d2a99e981c1.mp4 (885.5 kB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/4f8ee02d-1894-45bd-9519-2d1af275f9b2.mp4" autoplay muted></video>
<video controls width="19%" alt="videos_49_80ca7303ade985f9234e (1).mp4 (1.1 MB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/32ecb6eb-60fb-4f13-b176-ff86d7e11bf5.mp4" autoplay muted></video>
<video controls width="19%" alt="videos_49_b5c22f08e7ed2d27b6f7.mp4 (728.6 kB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/9f7b5f6a-dbe9-40a1-9420-cc4b6568b6c7.mp4" autoplay muted></video>
<figcaption>TD-MPC により学習したエージェント</figcaption>
</figure>
</div>

<div style="text-align: center">
<figure>
<video controls width="19%" alt="tdmpc2_cheetah.mp4 (404.5 kB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/a893b788-1d8d-4a4b-b79f-e72dc7c98759.mp4" autoplay muted></video>
<video controls width="19%" alt="videos_9_e6ebf64c7fef63e8ce4e.mp4 (1.5 MB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/e64bae8a-951b-4838-bbef-234e2052880b.mp4" autoplay muted></video>
<video controls width="19%" alt="videos_26_a069fa1b926540c77ba1.mp4 (877.9 kB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/428dc046-1e58-474e-abc2-33addce36931.mp4" autoplay muted></video>
<video controls width="19%" alt="videos_42_3051a376dc74b027ade5.mp4 (1.1 MB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/cd26c357-6475-43eb-995c-0df36ee63717.mp4" autoplay muted></video>
<video controls width="19%" alt="videos_49_b74d3abe0427b6fdefe1.mp4 (1.0 MB)" src="https://esa-storage-tokyo.s3-ap-northeast-1.amazonaws.com/uploads/production/attachments/21189/2024/04/01/160121/5b548125-1111-493d-b855-4ddf0423e5d4.mp4" autoplay muted></video>
<figcaption>TD-MPC2 により学習したエージェント</figcaption>
</figure>
</div>


## 環境設定
以下の手順に従って実行環境を設定します。

###  Dockerを使用したセットアップ

```
# イメージのビルド
docker build -t simple_tdmpc .

# コンテナの起動
docker run \
    --gpus all \
    -it \
    --rm \
    -w $HOME/work \
    -v $(pwd):$HOME/work \
    simple_tdmpc:latest bash
```

### 依存ライブラリのインストール
Poetryを使用して依存ライブラリをインストールします。

```
poetry install
```

## 実行方法

### TD-MPC の実行

    ```
    poetry run python src/tdmpc.py 
    ```
### TD-MPC2 の実行

    ```
    poetry run python src/tdmpc2.py 
    ```

## オプション

* `--capture_video` オプションを付与すると、学習プロセス中のビデオを `/videos` フォルダに保存できます。

    ```
    poetry run python src/tdmpc2.py --capture_video
    ```

*  `--track` オプションを付与すると、wandb を介して実験ログを記録できます。

    ```
    poetry run wandb login
    poetry run python src/tdmpc2.py --track --capture_video
    ```
    > 二行目で `transport faild error` が出る場合は、エラー文に出力される ```git config``` コマンドを実行して下さい。

*  `--task` オプションで学習するタスクを切り替えることができます。

    ```
    poetry run python src/tdmpc2.py --task 'dm_control/quadruped-run-v0' --total_timesteps 1000000
    ```


## 参照リポジトリ

- [TD-MPC](https://github.com/nicklashansen/tdmpc)
- [TD-MPC2](https://github.com/nicklashansen/tdmpc2)
- [CleanRL (Clean Implementation of RL Algorithms)](https://github.com/vwxyzjn/cleanrl)