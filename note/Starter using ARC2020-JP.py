# %% [markdown]
# # ARC2020ノートブックを使ったスターター 🏆📝
#
# ## はじめに
# ARC Prize 2024 提出ノートブックへようこそ！このノートブックは、ARC（Abstraction and Reasoning Corpus）Prize 2024 コンペティションへの提出プロセスを円滑にするために設計されています。このコンペティションでは、参加者は様々な抽象的な推論タスクを解くことができるアルゴリズムを開発することが求められます。
#
# ## 機能
# このノートブックには以下の機能があります：
# - 古いCSV形式から、ARC Prize 2024に必要な新しいJSON形式への提出の変換。
# - オブジェクト検出や操作など様々なテクニックを使ったARCタスクの解答予測。
# - コンペティションのガイドラインに沿った提出用CSVファイルの生成。
#
# ## ガイドライン
# このノートブックを効果的に使うために:
# 1. **データの準備**:
#    - 必要なJSONファイル（`arc-agi_test_challenges.json`）が指定のディレクトリ（`/kaggle/input/arc-prize-2024/`）にあることを確認してください。
#    - スクリプトは自動的に個別のタスクJSONファイルを保存するための`test`ディレクトリを作成します。
#
# **🌟 私のプロファイルや他の公開プロジェクトを見て、フィードバックをシェアするのを忘れないでください!**
#
# ## 👉 [私のプロフィールを見る]( https://www.kaggle.com/code/zulqarnainalipk) 👈
#
#
# 2. **提出プロセス**:
#    - 入力タスクに基づいて解答を予測する`predict_part_types`関数を実装・改良してください。
#    - オブジェクト検出（`get_objects_by_connectivity_`、`get_objects_by_color_`）などの手法を活用して予測を生成してください。
#    - 予測が提出用CSV（`sample_submission.csv`）に正しくフォーマットされていることを確認してください。
#
# 3. **実行**:
#    - `main()`関数を実行して提出プロセスを開始してください。
#    - 生成された`submission.json`ファイルをレビューして、提出の形式と内容を確認してください。
#
# ## 動作詳細
# - **データの読み込み**: JSONファイルが読み込まれ、処理のために個別のタスクファイルに分割されます。
# - **予測**: タスクは、検出されたオブジェクトとその属性に基づいて予測を生成するために`predict_part_types`のような関数を使って分析されます。
# - **提出**: 予測はCSVファイル（`old_submission.csv`）にまとめられ、`translate_submission`関数を使って必要なJSON形式に変換されます。
#
#
# ## 謝辞 🙏
# Abstraction and Reasoning Corpusオーガナイザーにデータセットとコンペティションプラットフォームを提供してくださったことに感謝します。
#
# さあ、始めましょう！途中で質問や支援が必要な場合は、遠慮なく連絡してください。
# 👉 [私のプロフィールを見る](https://www.kaggle.com/zulqarnainalipk) 👈
#

# %% [code]
# 📚 必要なライブラリのインポート 📊

import os  # OSに関連する操作用
import json  # JSON操作用
import csv  # CSVファイル操作用
import numpy as np  # NumPyは数値計算用
import pandas as pd  # Pandasはデータ操作用
from pathlib import Path  # Pathlibはパス操作用
import sys

# Kaggleで実行されているかチェックし、必要に応じてパスを調整
if os.path.exists('/kaggle/input'):
    sys.path.append('/kaggle/input/arclib4/')
else:
    sys.path.append('../input/arclib/')

# arclibから特定の関数とクラスをインポート 📦
from arclib.dsl import *
from arclib.util import evaluate_predict_func, get_string, data_path
from arclib.dsl import Task, unique_arrays  
from arclib.check import check_output_color_from_input


# %% [markdown]
# # 📂 JSONデータの読み込みと処理 📄
#

# %% [markdown]
# ### 概要
#
# このスクリプトは、指定されたファイルからJSONコンテンツを読み込み、'test'という名前のディレクトリが存在しない場合は作成し、読み込まれたJSONコンテンツを個別のファイルに分割します。JSONデータの各タスクは、'test'ディレクトリ内の個別のJSONファイルとして保存されます。
#
# ### ステップと機能
#
# 1. **JSONコンテンツの読み込み**
#    - `json_file_path`で指定されたファイルからJSONデータを開いて読み込みます。
#    - 読み込まれたJSONデータを`data`変数に格納して、さらに処理します。
#
# 2. **出力ディレクトリの作成**
#    - 個別のJSONファイルが保存されるパスを`output_dir`として定義します（デフォルトでは`'/kaggle/working/test'`）。
#    - `os.makedirs(output_dir, exist_ok=True)`を使用して、ディレクトリが存在しない場合は作成します（`exist_ok=True`により、ディレクトリが既に存在する場合にエラーが発生しないようにします）。
#
# 3. **JSONコンテンツの分割**
#    - 読み込まれたJSONデータ（`data.items()`）の各`task_id`と`task_data`のペアを反復処理します。
#    - `os.path.join(output_dir, f'{task_id}.json')`を使用して、各タスクの出力ファイルパスを構築します。
#    - 各出力ファイルを書き込みモード（`'w'`）で開き、対応する`task_data`を`json.dump(task_data, output_file, indent=4)`を使用してファイルに書き込み、可読性を向上させます。
#
# ### 使用方法
#
# - `json_file_path`が、JSONデータの正しい場所とファイル名を指していることを確認してください（この場合は`arc-agi_test_challenges.json`）。
# - スクリプトは、作業ディレクトリ（`/kaggle/working/`）に'test'という名前のディレクトリを作成し、JSONデータの各タスクを、そのディレクトリ内の個別のJSONファイルとして保存します。
# - このアプローチは、元々1つのJSONファイルに保存されている個々のタスクやデータセットを整理して管理するのに役立ちます。
#
# ---
#

# %% [code]
# JSONコンテンツを読み込む
json_file_path = '/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json'  # 🌐 実際のJSONファイルのパスに置き換えてください
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 'test'ディレクトリを作成する
output_dir = '/kaggle/working/test'  # 📁 必要に応じてこのパスを変更できます
os.makedirs(output_dir, exist_ok=True)

# JSONコンテンツを個別のファイルに分割する
for task_id, task_data in data.items():
    output_file_path = os.path.join(output_dir, f'{task_id}.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(task_data, output_file, indent=4)


# %% [markdown]
# # 📄 JSONコンテンツの読み込み
#

# %% [markdown]
# ### 概要
#
# このスクリプトは、提出データを含むJSONファイルを読み込み、各エントリを指定されたCSV形式に変換し、CSVファイルに書き込みます。CSVファイルは、`output_id`と`output`の2つの列で構成されています。ここで、`output`はJSONデータから派生したもので、2次元の行列をCSVに適した文字列形式に変換したものです。
#
# ### ステップと機能
#
# 1. **JSONデータの読み込み**
#    - スクリプトは、`json_file_path`で指定されたJSONファイルを開いて読み込むことから始まります。
#
# 2. **出力CSVファイルの定義**
#    - 変換されたデータが保存されるCSVファイルの出力パスを`output_csv_path`で定義します。
#
# 3. **2次元リストのCSV文字列形式への変換**
#    - 2次元リスト（行列）をCSVに適した文字列形式に変換する`convert_to_csv_format(matrix)`関数を定義します。各行の要素を'|'で結合し、'|'記号で囲みます。
#
# 4. **CSVファイルへの書き込み**
#    - `output_csv_path`を書き込みモード（`'w'`）で開き、CSVファイルにデータを書き込むための`csv.DictWriter`インスタンスを初期化します。
#    - `writer.writeheader()`を使用してヘッダー行（`'output_id', 'output'`）を書き込みます。
#
# 5. **各タスクの処理**
#    - 読み込まれたJSONデータ（`data.items()`）の各`task_id`と関連する`attempts`を反復処理します。
#    - 試行が1回だけの場合（`len(attempts) == 1`）、次の処理を行います。
#      - `output_id`を`{task_id}_0`の形式でフォーマットします。
#      - `convert_to_csv_format`を使用して、`attempt['attempt_1']`と`attempt['attempt_2']`をCSV形式に変換します。
#      - 両方の試行を`combined_output`に結合し、CSVファイルに書き込みます。
#    - 複数の試行がある場合は、各試行（`enumerate(attempts)`）を反復処理し、`output_id`をそれに応じてフォーマットします。
#
# 6. **完了メッセージ**
#    - 変換プロセスが正常に完了したことを示す完了メッセージ（`"✅ Sample submission CSV file created at {output_csv_path}"`）を表示します。
#
# ### 使用方法
#
# - `json_file_path`が、入力JSONファイル（この場合は`sample_submission.json`）の正しい場所とファイル名を指していることを確認してください。
# - 実行後、`sample_submission.csv`には、提出に適した指定の形式で処理された提出データが含まれます。
#
# ---
#

# %% [code]
json_file_path = '/kaggle/input/arc-prize-2024/sample_submission.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

# 出力CSVファイルのパスを定義する
output_csv_path = '/kaggle/working/sample_submission.csv'

# 2次元リストを必要な文字列形式に変換する関数
def convert_to_csv_format(matrix):
    return '|' + '|'.join(''.join(map(str, row)) for row in matrix) + '|'

# CSVファイルに処理して書き込む
with open(output_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['output_id', 'output']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for task_id, attempts in data.items():
        if len(attempts) == 1:
            attempt = attempts[0]
            output_id = f"{task_id}_0"
            attempt_1 = convert_to_csv_format(attempt['attempt_1'])
            attempt_2 = convert_to_csv_format(attempt['attempt_2'])
            combined_output = f"{attempt_1} {attempt_2}"
            writer.writerow({'output_id': output_id, 'output': combined_output})
        else:
            for i, attempt in enumerate(attempts):
                output_id = f"{task_id}_{i}"
                attempt_1 = convert_to_csv_format(attempt['attempt_1'])
                attempt_2 = convert_to_csv_format(attempt['attempt_2'])
                combined_output = f"{attempt_1} {attempt_2}"
                writer.writerow({'output_id': output_id, 'output': combined_output})

print(f"✅ サンプル提出用CSVファイルが{output_csv_path}に作成されました")


# %% [markdown]
# # 🔄 CSVからJSONに変換する関数
#

# %% [markdown]
#
# ### 概要
#
# この関数は、CSVファイルに保存されている古い提出形式を、ARC Prize 2024 コンペティションに必要な新しいJSON形式に変換します。CSVの各行を処理し、関連情報を抽出し、提出に適したJSON形式に構造化します。
#
# ### ステップと機能
#
# 1. **CSVファイルの読み込みと解析**
#    - 関数は、`file_path`で指定されたCSVファイルを開いて読み込むことから始まります。
#    - ヘッダー行を除く全ての行（`lines[1:]`）を読み込み、ヘッダー情報をスキップします。
#
# 2. **各行の処理**
#    - CSVファイルの各行について:
#      - `,`で行を分割して、`output_id`と`output`を抽出します。
#      - タスクと試行のインデックスを特定するために、`output_id`を`task_id`と`output_idx`に分割します。
#      - スペース`' '`に基づいて`output`を分割して、個々の予測を取得します。
#      - 2つ以上の予測がある場合は、最初の2つの予測のみを考慮します。
#      - 各予測を、'|'で区切られた文字列行（`pred_lines`）から整数を解析してマッピングすることにより、行列形式（`pred_matrix`）に変換します。
#
# 3. **試行の整理**
#    - 各試行（`attempt_dict`）を、`"attempt_1"`と`"attempt_2"`のキーが最初と2番目の試行の行列をそれぞれ表す辞書として構造化します。
#    - 試行を`submission_dict`の対応する`task_id`の下に挿入します。最初の試行（`output_idx == '0'`）の場合は先頭に挿入し（`insert(0, ...)`）、それ以外の場合は追加します。
#
# 4. **JSONファイルへの書き込み**
#    - 最後に、構造化された`submission_dict`を`'submission.json'`という名前の新しいJSONファイルに書き込みます。
#    - 可読性のために`indent=4`を使用して`json.dump()`を呼び出し、JSONにインデントを付けてフォーマットします。
#    
# 5. **確認メッセージ**
#    - 変換プロセスが正常に完了したことを示す確認メッセージ（`"✅ 変換された提出が 'submission.json' として保存されました"`）を表示します。
#
# ### 使用方法
#
# - CSVファイル（`file_path`）が、関数が期待する正しい形式になっていることを確認してください。
# - 実行後、`'submission.json'`には、ARC Prize 2024 コンペティションに提出できる、変換された提出データが含まれます。
#
# ---
#
#

# %% [code]
# 🔄 古い提出形式（csv）から新しい形式（json）に変換する関数

def translate_submission(file_path):
    # 元の提出ファイルを読み込む
    with open(file_path, 'r') as file:
        lines = file.readlines()

    submission_dict = {}

    for line in lines[1:]:  # ヘッダー行をスキップする
        output_id, output = line.strip().split(',')
        task_id, output_idx = output_id.split('_')
        predictions = output.split(' ')  # ' 'に基づいて予測を分割する
        
        # 最初の2つの予測のみを取る
        if len(predictions) > 2:
            predictions = predictions[:2]

        processed_predictions = []
        for pred in predictions:
            if pred:  # predが空の文字列でないことを確認する
                pred_lines = pred.split('|')[1:-1]  # 分割から空の文字列を削除する
                pred_matrix = [list(map(int, line)) for line in pred_lines]
                processed_predictions.append(pred_matrix)

        attempt_1 = processed_predictions[0] if len(processed_predictions) > 0 else []  # 試行1の行列
        attempt_2 = processed_predictions[1] if len(processed_predictions) > 1 else []  # 試行2の行列

        if task_id not in submission_dict:
            submission_dict[task_id] = []

        attempt_dict = {
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        }

        if output_idx == '0':
            submission_dict[task_id].insert(0, attempt_dict)  # 最初の試行を挿入する
        else:
            submission_dict[task_id].append(attempt_dict)  # 後続の試行を追加する
    
    # 新しいJSONファイルに書き込む
    with open('submission.json', 'w') as file:
        json.dump(submission_dict, file, indent=4)  # 可読性のためにインデントを付けてJSONを保存する

    print(f"✅ 変換された提出が 'submission.json' として保存されました")  # 変換成功後の確認メッセージ



# %% [markdown]
# # タスク提出と予測スクリプト🧩

# %% [markdown]
#
#
#
# #### 1. ファイルポインタとパス
#
# ```python
# data_path = Path('/kaggle/working/')  # 📁 作業ディレクトリへのパス
# test_path = data_path / 'test'         # 📁 テストディレクトリへのパス
# sample_submission_path = '/kaggle/working/sample_submission.csv'  # 📄 サンプル提出CSVへのパス
# ```
#
# - **説明**: 
#   - `data_path`と`test_path`: 入力データ（タスク）とテストファイルがそれぞれ保存されるディレクトリを定義します。
#   - `sample_submission_path`: 予測がフォーマットされ、書き込まれるサンプル提出CSVファイルを指します。
#
# #### 2. 候補に対する出力のチェック
#
# ```python
# def check_output_in_candidates(output, candidates):
#     """
#     与えられた出力が候補出力のいずれかと一致するかどうかを確認します。
#     """
#     output_is_candidate = False
#
#     for candidate in candidates:
#         if output.shape == candidate.shape:
#             if (output == candidate).all():
#                 output_is_candidate = True
#                 break
#     return output_is_candidate
# ```
#
# - **説明**: 
#   - `check_output_in_candidates`: 予測された出力が候補解のいずれかと一致するかを検証する関数。これは予測中の検証に重要です。
#
# #### 3. 予測関数
#
# ```python
# def predict_part(task, get_candidates, train_object_maps=None, train_bg_colors=None):
#     """
#     候補生成を使用して、入力-出力ペアに基づいてタスクの解を予測します。
#     """
#     # 上記のコードで説明されている実装の詳細...
# ```
#
# - **説明**: 
#   - `predict_part`: 解を予測するための中心的な関数。入力-出力ペアを反復処理し、候補を生成し、`check_output_in_candidates`を使用して期待される出力に対して検証します。
#
# #### 4. オブジェクト操作関数
#
# ```python
# def get_cropped_object(array, object_map, bg_color=None):
#     """
#     与えられたオブジェクトマップに基づいて、配列のクロップされた部分を取得します。
#     """
#     # 上記のコードで説明されている実装の詳細...
# ```
#
# - **説明**: 
#   - `get_cropped_object`, `keep_one_object`, `get_cropped_objects`, `get_inputs_with_one_object`のような関数は、提供されたオブジェクトマップに基づいてオブジェクトを抽出するための配列操作のユーティリティ関数です。
#
# #### 5. 予測タイプと戦略
#
# ```python
# def predict_part_types(task):
#     """
#     入力の特性と候補生成に基づいて、タスクの解のタイプを予測します。
#     """
#     # 上記のコードで説明されている実装の詳細...
# ```
#
# - **説明**: 
#   - `predict_part_types`: 入力の特性と候補生成方法に基づいて、解のタイプを予測するために異なる戦略（`get_object_map_funcs`）を使用します。
#
# #### 6. 提出と結果の処理
#
# ```python
# def submit(predict):
#     """
#     テストタスクの予測を提出し、新しい提出CSVファイルを作成します。
#     """
#     # 上記のコードで説明されている実装の詳細...
# ```
#
# - **説明**: 
#   - `submit`: テストタスクを読み込み、`predict`を使用して出力を予測し、コンペティションの要件に従ってフォーマットし、`old_submission.csv`に出力します。これは、`translate_submission`を使用して最終的な提出形式に変換されます。
#
# ### 実行の流れ
#
# - **メイン実行**: スクリプトが実行されると、`main()`関数が呼び出され、それが`submit(predict_part_types)`を呼び出します。
# - **提出プロセス**: 
#   - サンプル提出CSVを読み込みます。
#   - テストファイルを反復処理し、`predict_part_types`を使用して出力を予測します。
#   - 予測をフォーマットし、`old_submission.csv`に保存します。
#   - CSV提出を、ARC Prize 2024に必要なJSON形式に変換します。
#
#
#

# %% [code]
# 📂 上記で作成したARC Prize JSONsからファイルとサンプル提出CSVへのポインタを追加する

data_path = Path('/kaggle/working/')  # 📁 作業ディレクトリへのパス
test_path = data_path / 'test'  # 📁 テストディレクトリへのパス

sample_submission_path = '/kaggle/working/sample_submission.csv'  # 📄 サンプル提出CSVへのパス

# 出力が候補のいずれかと一致するかどうかを確認する関数
def check_output_in_candidates(output, candidates):
    """
    与えられた出力が候補出力のいずれかと一致するかどうかを確認します。

    Args:
    - output (numpy.ndarray): 比較する出力行列。
    - candidates (numpy.ndarrayのリスト): 比較する候補行列のリスト。

    Returns:
    - output_is_candidate (bool): 出力が候補のいずれかと一致する場合はTrue、そうでない場合はFalse。
    """
    output_is_candidate = False

    for candidate in candidates:
        if output.shape == candidate.shape:
            if (output == candidate).all():
                output_is_candidate = True
                break
    return output_is_candidate


# タスクの入力と候補に基づいて予測する関数
def predict_part(task, get_candidates, train_object_maps=None, train_bg_colors=None):
    """
    候補生成を使用して、入力-出力ペアに基づいてタスクの解を予測します。

    Args:
    - task (Task): 入力-出力ペアとテスト入力を含むタスクオブジェクト。
    - get_candidates (function): 入力とオブジェクトマップに基づいて候補を生成する関数。
    - train_object_maps (numpy.ndarrayのリスト, optional): トレーニング入力のオブジェクトマップ。
    - train_bg_colors (list, optional): トレーニング入力の背景色。

    Returns:
    - all_input_predictions (list): タスクが完全に解ける場合は各テスト入力の予測のリスト、そうでない場合は空のリスト。
    """
    part_task = True
    for i, (input, output) in enumerate(task.pairs):
        input = np.array(input)
        output = np.array(output)

        candidates = get_candidates(input, object_maps=train_object_maps[i], bg_color=train_bg_colors[i])

        if candidates:
            if not check_output_in_candidates(output, candidates):
                part_task = False
                break
        else:
            part_task = False
            break

    all_input_predictions = []
    if part_task:
        for input in task.test_inputs:
            test_candidates = get_candidates(input)
            predictions = test_candidates
            predictions = unique_arrays(predictions)
            predictions = sorted(predictions, key=lambda x: x.shape[0] * x.shape[1], reverse=True)
            all_input_predictions.append(predictions)

    else:
        all_input_predictions = []

    return all_input_predictions


# オブジェクトマップに基づいてクロップされたオブジェクトを取得する関数
def get_cropped_object(array, object_map, bg_color=None):
    """
    与えられたオブジェクトマップに基づいて、配列のクロップされた部分を取得します。

    Args:
    - array (numpy.ndarray): クロップされたオブジェクトを取得する入力配列。
    - object_map (numpy.ndarray): オブジェクトの位置を示すバイナリマップ。
    - bg_color (int or float, optional): オブジェクトの境界の外側を埋める背景色。

    Returns:
    - cropped_object (numpy.ndarray): 入力配列からクロップされたオブジェクト。
    """
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    return array[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]


# オブジェクトマップに基づいて1つのオブジェクトを保持する関数
def keep_one_object(array, object_map, bg_color=None):
    """
    与えられたオブジェクトマップに基づいて、入力配列から1つのオブジェクトのみを抽出して保持します。

    Args:
    - array (numpy.ndarray): 複数のオブジェクトを含む入力配列。
    - object_map (numpy.ndarray): 保持するオブジェクトの位置を示すバイナリマップ。
    - bg_color (int or float, optional): オブジェクトの境界の外側を埋める背景色。

    Returns:
    - retained_object (numpy.ndarray): 指定されたオブジェクトのみが保持された配列。背景が指定されている場合は背景が塗りつぶされます。
    """
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    if bg_color is None:
        bg_color = detect_bg_(array)
    output_ = np.full_like(array, bg_color)
    output_[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1] = array[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]
    return output_


# オブジェクトマップに基づいてクロップされたオブジェクトを取得する関数
def get_cropped_objects(array, get_object_maps=None, object_maps=None, augment=None, bg_color=None):
    """
    検出されたオブジェクトマップに基づいて、入力配列からクロップされたオブジェクトを取得します。

    Args:
    - array (numpy.ndarray): オブジェクトを取得する入力配列。
    - get_object_maps (function, optional): 配列内のオブジェクトマップを検出する関数。
    - object_maps (numpy.ndarrayのリスト, optional): 事前に検出されたオブジェクトマップ。
    - augment (function, optional): クロップされたオブジェクトに適用する拡張関数。
    - bg_color (int or float, optional): オブジェクトの境界の外側を埋める背景色。

    Returns:
    - objects (numpy.ndarrayのリスト): 入力配列から抽出されたクロップされたオブジェクトのリスト。
    """
    if object_maps is None:
        object_maps = get_object_maps(array)
    objects = [augment(get_cropped_object(array, object_map)) for object_map in object_maps if np.count_nonzero(object_map) > 0]
    return objects


# オブジェクトマップに基づいて1つのオブジェクトを含む入力を取得する関数
def get_inputs_with_one_object(array, get_object_maps=None, object_maps=None, augment=None, bg_color=None):
    """
    検出されたオブジェクトマップに基づいて、正確に1つのオブジェクトを含む入力を取得します。

    Args:
    - array (numpy.ndarray): 1つのオブジェクトを含む入力を取得する入力配列。
    - get_object_maps (function, optional): 配列内のオブジェクトマップを検出する関数。
    - object_maps (numpy.ndarrayのリスト, optional): 事前に検出されたオブジェクトマップ。
    - augment (function, optional): 保持されたオブジェクトに適用する拡張関数。
    - bg_color (int or float, optional): オブジェクトの境界の外側を埋める背景色。

    Returns:
    - objects (numpy.ndarrayのリスト): 正確に1つのオブジェクトを含む入力のリスト。
    """
    if object_maps is None:
        object_maps = get_object_maps(array)
    objects = [augment(keep_one_object(array, object_map, bg_color=bg_color)) for object_map in object_maps if np.count_nonzero(object_map) > 0]
    return objects


# オブジェクトマップを取得する関数のリスト
get_object_map_funcs = [get_objects_by_connectivity_, partial(get_objects_by_connectivity_, touch='corner'),
                        get_objects_by_color_and_connectivity_, partial(get_objects_by_color_and_connectivity_, touch='corner'), get_objects_by_color_,
                        get_objects_rectangles, partial(get_objects_rectangles, direction='horisontal'), get_objects_rectangles_without_noise, get_objects_rectangles_without_noise_without_padding]


# タスクの入力に基づいてパートタイプを予測する関数
def predict_part_types(task):
    """
    入力の特性と候補生成に基づいて、タスクの解のタイプを予測します。

    Args:
    - task (Task): 解を予測する入力を含むタスクオブジェクト。

    Returns:
    - predictions (list): 解ける場合は各入力の予測解のリスト、そうでない場合は空のリスト。
    """
    predictions = []

    if check_output_color_from_input(task):
        bg_colors = [detect_bg_(input_) for input_ in task.inputs]
        for i, get_object_maps in enumerate(get_object_map_funcs):

            object_maps_list = [get_object_maps(input_, bg_color=bg_color) for input_, bg_color in zip(task.inputs, bg_colors)]
            for get_object_func in (get_cropped_objects, get_inputs_with_one_object):
                for augment in simple_output_process_options:
                    get_candidates = partial(get_object_func, get_object_maps=get_object_maps, augment=augment)
                    predictions = predict_part(task, get_candidates=get_candidates, train_object_maps=object_maps_list, train_bg_colors=bg_colors)
                    if predictions:
                        break
                if predictions:
                    break
            if predictions:
                break

    return predictions


# 予測を提出し、新しい提出CSVファイルを作成する関数
def submit(predict):
    """
    テストタスクの予測を提出し、新しい提出CSVファイルを作成します。

    Args:
    - predict (function): タスクの解を予測する関数。

    Returns:
    - None
    """
    submission = pd.read_csv(sample_submission_path, index_col='output_id')
    submission['output'] = ''
    test_fns = sorted(os.listdir(test_path))
    count = 0
    for fn in test_fns:
        fp = test_path / fn
        with open(fp, 'r') as f:
            task = Task(json.load(f))
            all_input_preds = predict(task)
            if all_input_preds:
                print(fn)
                count += 1

            for i, preds in enumerate(all_input_preds):
                output_id = str(fn.split('.')[-2]) + '_' + str(i)
                string_preds = [get_string(pred) for pred in preds[:3]]
                pred = ' '.join(string_preds)
                submission.loc[output_id, 'output'] = pred

    print(count)
    submission.to_csv('old_submission.csv')  # 提出をCSVに保存
    # ARC Prize 2024のガイドラインに従った提出ファイルを作成
    translate_submission('/kaggle/working/old_submission.csv')



# %% [code]
# 提出プロセスを実行するメイン関数
def main():
    submit(predict_part_types)


if __name__ == '__main__':
    main()


# %% [markdown]
# ## 🌟 探求を続けよう! 🌟
#
# このノートブックを深く探求していただきありがとうございます！楽しんでいただけたり、新しいことを学んでいただけたら、私のプロフィールにある他の魅力的なプロジェクトや貢献にも飛び込んでみてください。
#
# 👉 [もっと探求しましょう！](https://www.kaggle.com/zulqarnainalipk) 👈
#
# [GitHub](https://github.com/zulqarnainalipk) |
# [LinkedIn](https://www.linkedin.com/in/zulqarnainalipk/)
#
# ## 💬 あなたの考えをシェアしてください！ 💡
#
# あなたのフィードバックは私たちにとって宝物のようなものです！あなたの素晴らしいアイデアと洞察は、私たちの継続的な改善の原動力となります。言いたいこと、尋ねたいこと、提案したいことがあれば、遠慮なく！
#
# 📬 Eメールでご連絡ください：[zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)
#
# あなたの時間と関わりに大変感謝しています。あなたのサポートは、私がさらにエピックなコンテンツを作成するための推進力のようなものです。
# 楽しくコーディングを続け、データサイエンスの冒険で素晴らしい成功を収めることを願っています！ 🚀
#
