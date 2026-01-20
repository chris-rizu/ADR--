---

# GCPベース AI活用型著作権侵害検知システム

## システムアーキテクチャ概要

### ハイレベルアーキテクチャ

```text
┌─────────────────────────────────────────────────────────────┐
│                     インジェッション（取込）層                │
│  • Cloud Storage アップロード (署名付きURL)                  │
│  • Cloud Pub/Sub (メッセージキュー)                          │
│  • 入力検証と前処理                                          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                      特徴量抽出層                            │
│  ┌──────────────────┐         ┌─────────────────┐          │
│  │ 視覚的パイプライン  │         │  OCRパイプライン   │          │
│  │  • 知覚的ハッシュ   │         │  • Cloud Vision │          │
│  │    (Perceptual   │         │    API          │          │
│  │     Hashing)     │         │  • Document AI  │          │
│  │  • Vertex AI     │         │  • テキスト分析     │          │
│  │    マルチモーダル   │         │                 │          │
│  │    埋め込み       │         │                 │          │
│  │  • 特徴ベクトル     │         │                 │          │
│  └──────────────────┘         └─────────────────┘          │
└────────────┬──────────────────────────┬────────────────────┘
             │                          │
             ▼                          ▼
┌─────────────────────────┐  ┌──────────────────────────────┐
│   類似性マッチング        │  │      著作権テキスト分析        │
│  • Vertex AI Vector     │  │  • パターンマッチング           │
│  •   Search (Matching   │  │  • エンティティ認識             │
│      Engine)            │  │  • 年/出版社抽出               │
│  • 近似最近傍探索 (ANN)   │  │  • Natural Language API      │
│  • マルチレベルマッチング  │  │                              │
└────────────┬────────────┘  └─────────────┬────────────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      判定エンジン                            │
│  • スコア集計と重み付け                                      │
│  • 閾値ベースの分類                                          │
│  • 説明可能性 (Explainability) 生成                          │
│  • Cloud Functions (オーケストレーション)                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                       出力層                                 │
│  • マッチング結果 (JSON/Protobuf)                            │
│  • 信頼度スコアと証拠データ                                   │
│  • 監査ログ (Cloud Logging)                                  │
│  • 分析 (BigQuery)                                          │
└─────────────────────────────────────────────────────────────┘

```

## 詳細コンポーネント設計

### 1. インジェッション（取込）層

**目的:** 画像を受け入れ、検証、前処理を行い、GCPネイティブサービスを使用して処理パイプラインへルーティングする。

**コンポーネント:**

* **A. Cloud Storage (入力バケット)**
* **構成:**
* 高可用性のためのマルチリージョンバケット
* オブジェクトライフサイクル管理（処理済み画像は30日後に自動削除）
* IAMによる均一なバケットレベルアクセス
* 監査証跡のためにオブジェクトのバージョニングを有効化




* **B. Cloud Storage トリガー + Cloud Pub/Sub**
* **イベントフロー:** 画像アップロード → Cloud Storage トリガー → Pub/Sub メッセージ → Cloud Run/Functions
* **Pub/Sub トピック:**
* `image-upload-events`: 新規画像通知
* `batch-processing-queue`: 一括処理リクエスト
* `priority-queue`: 優先度の高いリアルタイムチェック
* `retry-queue`: 指数バックオフを伴う処理失敗時のリトライ




* **C. API Gateway (Cloud Endpoints + Cloud Run)**
* **エンドポイント:**
* `POST /api/v1/check/image` - 単一画像のアップロード
* `POST /api/v1/check/batch` - バッチアップロード（ジョブIDを返す）
* `GET /api/v1/status/{job_id}` - 処理状況の確認
* `GET /api/v1/results/{job_id}` - 結果の取得




* **D. 前処理サービス (Cloud Functions Gen2)**

```python
# Cloud Functionとしてデプロイ
import functions_framework
from google.cloud import storage
from PIL import Image
import io

@functions_framework.cloud_event
def preprocess_image(cloud_event):
    """
    Cloud Storageへのアップロードによってトリガーされる
    - 画像形式の検証
    - 寸法の正規化 (最大2048px)
    - 標準フォーマット (JPEG) への変換
    - 処理キューへのパブリッシュ
    """
    bucket_name = cloud_event.data["bucket"]
    file_name = cloud_event.data["name"]
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # ダウンロードと検証
    image_bytes = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_bytes))
    
    # 検証
    if image.format not in ['JPEG', 'PNG', 'WEBP']:
        raise ValueError(f"Unsupported format: {image.format}")
    
    # 寸法の正規化 (アスペクト比を保持)
    max_dimension = 2048
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    
    # 必要に応じてRGBへ変換
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 正規化されたバージョンを保存
    output_buffer = io.BytesIO()
    image.save(output_buffer, format='JPEG', quality=95)
    
    # 処理用バケットへアップロード
    processed_blob = bucket.blob(f"processed/{file_name}")
    processed_blob.upload_from_string(
        output_buffer.getvalue(),
        content_type='image/jpeg'
    )
    
    # ダウンストリーム処理のためにPub/Subへパブリッシュ
    publish_to_processing_queue({
        'image_path': f"gs://{bucket_name}/processed/{file_name}",
        'original_path': f"gs://{bucket_name}/{file_name}",
        'dimensions': image.size,
        'format': 'JPEG'
    })

```

**フロー:**
ユーザーアップロード → 署名付きURL → Cloud Storage → ストレージトリガー
→ 前処理Function → 検証 → Pub/Sub → 処理パイプライン

---

### 2. 視覚的類似性パイプライン

これは、すべてGCPネイティブの技術を用いた、複数の補完的な技術を使用するコアとなるマッチングエンジンです。

#### A. 知覚的ハッシュ (Perceptual Hashing - 高速な近似マッチング)

**GCPでの実装:**

```python
# Cloud Runサービスとしてデプロイ
from flask import Flask, request, jsonify
from google.cloud import storage, firestore
from imagehash import phash, dhash, whash, average_hash
from PIL import Image
import io

app = Flask(__name__)
db = firestore.Client()

@app.route('/hash', methods=['POST'])
def generate_hashes():
    """
    堅牢性のために複数の知覚的ハッシュを生成する
    """
    data = request.json
    image_path = data['image_path']
    
    # Cloud Storageからダウンロード
    storage_client = storage.Client()
    bucket_name, blob_name = parse_gcs_path(image_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    image_bytes = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_bytes))
    
    # 冗長性のために複数のハッシュを生成
    hashes = {
        'phash': str(phash(image, hash_size=16)),      # 16x16 = 256 bits
        'dhash': str(dhash(image, hash_size=16)),
        'whash': str(whash(image, hash_size=16)),
        'average_hash': str(average_hash(image, hash_size=16))
    }
    
    return jsonify({
        'hashes': hashes,
        'image_path': image_path
    })

def find_similar_by_hash(query_hashes, threshold=10):
    """
    ハミング距離を使用してFirestoreから類似ハッシュをクエリする
    閾値: 最大ビット差 (16x16ハッシュの場合は0-256)
    """
    results = []
    
    # リファレンスデータベースに対してクエリを実行
    # 注意: Firestoreはハミング距離をネイティブでサポートしていないため、
    # 候補を取得してメモリ内で計算する
    
    ref_docs = db.collection('perceptual_hashes').stream()
    
    for doc in ref_docs:
        ref_hashes = doc.to_dict()
        
        # 各ハッシュタイプについてハミング距離を計算
        distances = {
            'phash': hamming_distance(query_hashes['phash'], ref_hashes['phash']),
            'dhash': hamming_distance(query_hashes['dhash'], ref_hashes['dhash']),
            'whash': hamming_distance(query_hashes['whash'], ref_hashes['whash']),
        }
        
        # いずれかのハッシュが閾値を下回れば、マッチとみなす
        min_distance = min(distances.values())
        
        if min_distance <= threshold:
            results.append({
                'ref_id': doc.id,
                'distances': distances,
                'min_distance': min_distance,
                'match_type': 'perceptual_hash'
            })
    
    # 最小距離でソート
    results.sort(key=lambda x: x['min_distance'])
    return results[:10]  # 上位10件の一致

def hamming_distance(hash1, hash2):
    """2つのハッシュ間のハミング距離を計算"""
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')

```

**ストレージ戦略:**

* **Firestore:** ハッシュ値をメタデータと共に保存
* コレクション: `perceptual_hashes`
* ドキュメント構造: `{ "image_id": "copyright_img_12345", "phash": "...", "publisher": "PublisherX", ... }`



**大規模化への最適化:**

* ハッシュ参照キャッシュに **Memorystore (Redis)** を使用
* RedisにBK木構造を実装し、準線形時間のハミング距離クエリを実現
* 分散クエリ用にハッシュプレフィックスでパーティショニング

#### B. ディープラーニング埋め込み (意味的類似性)

**Vertex AI Multimodal Embeddings API:**

```python
# 埋め込み生成用 Cloud Run サービス
from google.cloud import aiplatform
from google.cloud import storage
import base64

aiplatform.init(project='your-project-id', location='us-central1')

def generate_multimodal_embedding(image_gcs_path):
    """
    Vertex AI Multimodal Embeddings を使用して埋め込みを生成
    サポート次元: 128, 256, 512, 1408
    """
    
    # 画像の読み込み処理... (省略)
    
    # Vertex AI Multimodal Embeddings API の呼び出し
    from vertexai.vision_models import MultiModalEmbeddingModel
    
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    
    embeddings = model.get_embeddings(
        image=aiplatform.gapic.types.Image(image_bytes=image_bytes),
        dimension=1408  # 最高品質
    )
    
    return {
        'embedding': embeddings.image_embedding,
        'dimension': 1408,
        'model': 'multimodalembedding@001'
    }

# 代替案: GKE上のセルフホストCLIP
def generate_clip_embedding_gke(image_bytes):
    """
    コスト最適化またはカスタムモデルの場合、
    GPUノードを持つGKE上にCLIPをデプロイする
    """
    # (コード省略: PyTorchとTransformersを使用)
    pass

```

**モデル選択マトリックス:**

| ユースケース | 推奨モデル | 次元数 | デプロイメント |
| --- | --- | --- | --- |
| **本番環境 (マネージド)** | Vertex AI Multimodal | 1408 | フルマネージドAPI |
| **コスト最適化** | Vertex AI Multimodal | 512 | マネージドAPI |
| **カスタムファインチューニング** | CLIP ViT-L/14 on GKE | 768 | GKE GPUノード |
| **超低レイテンシ** | CLIP ViT-B/32 on GKE | 512 | GKE GPUノード |
| **最高精度** | DINOv2 ViT-g/14 on GKE | 1536 | GKE GPUノード |

#### C. Vertex AI Vector Search (Matching Engine)

**インデックス作成と設定:**

```python
def create_vector_search_index():
    """
    効率的な類似性検索のためにVertex AI Vector Searchインデックスを作成
    """
    # ... (クライアント初期化)
    
    # インデックス設定の定義
    index_config = {
        "display_name": "copyright-image-embeddings",
        "description": "著作権検知用ベクトル検索インデックス",
        "metadata": {
            "config": {
                "dimensions": 1408,
                "approximate_neighbors_count": 150,
                "distance_measure_type": "COSINE_DISTANCE", # コサイン類似度
                "algorithm_config": {
                    "tree_ah_config": {
                        "leaf_node_embedding_count": 500,
                        "leaf_nodes_to_search_percent": 7
                    }
                },
                "shard_size": "SHARD_SIZE_SMALL"
            }
        },
        "index_update_method": "STREAM_UPDATE"  # リアルタイム更新
    }
    # ... (作成処理)

def search_similar_images(query_embedding, top_k=10):
    """
    Vertex AI Vector Searchで類似画像をクエリ
    """
    # ... (リクエスト作成とレスポンス処理)
    
    # 結果のパース
    matches = []
    for neighbor in response.nearest_neighbors[0].neighbors:
        matches.append({
            'id': neighbor.datapoint.datapoint_id,
            'distance': neighbor.distance,
            'similarity_score': 1 - neighbor.distance  # 距離を類似度に変換
        })
    
    return matches

```

**バッチインデックスパイプライン:**

* `batch_index_reference_images()`: 著作権付き参照画像をバッチ処理し、Vector Searchに追加する。
* `upload_to_vector_search()`: Vertex AI Vector Searchへのストリーム更新を行う。

**パフォーマンス特性:**

* **クエリレイテンシ:** 1000万ベクトルで50-100ms (p95)
* **スループット:** エンドポイントあたり 1000+ QPS
* **インデックス更新:** リアルタイムストリーミング更新 (< 1分で反映)
* **スケーラビリティ:** 自動シャーディングにより数十億ベクトルまで対応

---

### 3. OCRパイプライン

GCPサービスを用いたマルチエンジンアプローチ:

#### A. Cloud Vision API (プライマリOCR)

```python
def detect_copyright_text_vision_api(image_gcs_path):
    """
    包括的なテキスト検出のためにCloud Vision APIを使用
    """
    # DOCUMENT_TEXT_DETECTION を使用して高密度テキストに対応
    # ... (API呼び出しとテキスト抽出)
    pass

def analyze_copyright_patterns(ocr_result):
    """
    著作権インジケーターのパターンマッチング
    """
    # 正規表現パターン
    patterns = {
        'copyright_symbol': r'[©Ⓒⓒ℗®™]',          # 著作権記号
        'copyright_word': r'\b[Cc]opyright\b',     # "Copyright" 単語
        'rights_reserved': r'\b[Aa]ll\s+[Rr]ights\s+[Rr]eserved\b',
        'year_basic': r'\b(19|20)\d{2}\b',         # 年号
        # ... (出版社パターンなど)
    }
    
    # ... (マッチング処理と信頼度計算)

```

#### B. Document AI (複雑なドキュメント向け)

* PDFや複数ページのドキュメント内の複雑な著作権表示に対して使用。
* `copyright_year`、`publisher`などのエンティティを認識するようにトレーニング可能。

#### C. Natural Language API (エンティティおよびセンチメント分析)

* コンテキストを理解し、テキストから組織名（Organization）や人名（Person）などのエンティティを抽出するために使用。

**OCRパイプライン・オーケストレーション (Cloud Workflows):**
`ocr_vision` (視覚) → `analyze_patterns` (パターン分析) → `entity_extraction` (エンティティ抽出) → 結果の統合

---

### 4. 判定エンジン

包括的なスコアリングシステム (Cloud Functions):

```python
@functions_framework.http
def calculate_copyright_score(request):
    """
    すべての分析結果を結合するメイン判定エンジン
    """
    # ...
    
    # 1. 視覚的類似性スコア (0-100)
    visual_score = calculate_visual_score(visual_results)
    
    # 2. OCR著作権インジケータースコア (0-100)
    ocr_score = calculate_ocr_score(ocr_results)
    
    # 3. メタデータ整合性スコア (0-100)
    metadata_score = calculate_metadata_score(...)
    
    # 4. 重み付けされた最終スコア
    weights = {
        'visual': 0.60,      # 視覚的一致が主要なシグナル
        'ocr': 0.30,         # 著作権テキストは強力なインジケーター
        'metadata': 0.10     # メタデータ整合性チェック
    }
    
    final_score = (...)
    
    # 5. リスク分類
    classification = classify_risk(final_score, visual_results, ocr_results)
    
    # 6. 説明文の生成 (Explainability)
    explanation = generate_explanation(...)
    
    # 7. 証拠のコンパイル
    evidence = compile_evidence(...)
    
    # ... (Firestoreへの保存とJSON返却)

```

* **calculate_visual_score:** ベクトル類似度とハッシュ距離に基づいてスコアリング。
* **calculate_ocr_score:** ©記号、年号、所有者情報などが揃っている場合にボーナス点を付与。
* **classify_risk:** スコアに基づき `CRITICAL` (要即時確認) から `MINIMAL` (アクション不要) まで分類。
* **generate_explanation:** 「画像は著作権付き参照画像Xと95%の視覚的類似性があります」といった人間が読める説明を生成。

---

### 5. オーケストレーションとワークフロー

**Cloud Workflows 定義:**
メインフローは以下のステップを実行します：

1. **初期化:** 入力パラメータの割り当て。
2. **並列処理:**
* **視覚パイプライン:** ハッシュ生成 → 埋め込み生成 → 類似検索。
* **OCRパイプライン:** OCR検出 → 著作権分析。


3. **判定エンジン:** 両パイプラインの結果を集約して判定。
4. **保存と通知:** Firestoreへ保存し、重要度が高い場合はWebhookでアラート送信。

---

## クラウドデプロイメントアーキテクチャ

### GCPインフラストラクチャスタック

```text
┌──────────────────────────────────────────────────────────────┐
│                    Cloud Load Balancing                       │
│              (グローバル HTTPS ロードバランサ)                 │
└──────────────┬──────────────────────────────────────────────┬┘
               │                                                 │
               ▼                                                 ▼
┌──────────────────────────┐                     ┌────────────────────────┐
│   Cloud Run (API)        │                     │  Cloud Run (ワーカー)   │
│  • FastAPI サービス       │                     │  • 埋め込み生成         │
│  • オートスケール 0-1000   │                     │  • ハッシュ処理         │
│  • CPU: 4 vCPU           │                     │  • GPU: T4 (オプション) │
│  • メモリ: 8GB            │                     │  • オートスケール 0-100  │
└────────┬─────────────────┘                     └───────────┬────────────┘
         │                                                    │
         ▼                                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                      Cloud Pub/Sub                            │
│  トピック:                                                     │
│    • image-uploads (メインキュー)                             │
│    • priority-checks (リアルタイム)                           │
│    • batch-processing (一括ジョブ)                            │
│    • results-notifications                                   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                       処理層                                  │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Cloud Functions│  │ GKEワークロード│  │ Cloud Workflows │  │
│  │  • 前処理       │  │  • GPUノード  │  │  • オーケスト   │  │
│  │  • バリデータ   │  │  • CLIPモデル │  │    レーション   │  │
│  └────────────────┘  └──────────────┘  └─────────────────┘  │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                    GCP AI/ML サービス                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Vertex AI                                             │   │
│  │  • Multimodal Embeddings API                         │   │
│  │  • Vector Search (Matching Engine)                   │   │
│  │  • Prediction Endpoints                              │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Cloud Vision API                                      │   │
│  │  • ドキュメントテキスト検出                               │   │
│  │  • ロゴ検出 (オプション)                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Document AI (オプション)                               │   │
│  │  • 複雑なドキュメント処理                                 │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                     データストレージ層                        │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Cloud Storage  │  │  Firestore   │  │  Memorystore    │  │
│  │  • 入力         │  │  • メタデータ │  │  (Redis)        │  │
│  │  • リファレンス  │  │  • 結果       │  │  • ハッシュ      │  │
│  │  • サムネイル    │  │  • ジョブ     │  │    キャッシュ    │  │
│  └────────────────┘  └──────────────┘  └─────────────────┘  │
│  ┌────────────────┐  ┌──────────────┐                        │
│  │  BigQuery      │  │  Cloud SQL   │                        │
│  │  • 分析         │  │  • 監査ログ   │                        │
│  │  • レポーティング│  │  • ユーザー   │                        │
│  └────────────────┘  └──────────────┘                        │
└───────────────────────────────────────────────────────────────┘

```

### 技術スタック概要

| コンポーネント | GCPサービス | 構成 | 代替案 |
| --- | --- | --- | --- |
| **APIゲートウェイ** | Cloud Run | 4 vCPU, 8GB RAM | Cloud Functions |
| **メッセージキュー** | Cloud Pub/Sub | Standard tier | - |
| **画像ストレージ** | Cloud Storage | Multi-regional | - |
| **知覚的ハッシュ** | Cloud Run | Custom Python | Cloud Functions |
| **ディープラーニング** | Vertex AI Embeddings | 1408次元 | GKE + CLIP |
| **ベクトルデータベース** | Vertex AI Vector Search | COSINE, ANN | Self-hosted Milvus on GKE |
| **OCRエンジン** | Cloud Vision API | DOCUMENT_TEXT | Document AI |
| **テキスト分析** | Natural Language API | エンティティ抽出 | - |
| **オーケストレーション** | Cloud Workflows | YAML定義 | Cloud Composer |
| **メタデータDB** | Firestore | Native mode | Cloud SQL PostgreSQL |
| **キャッシング** | Memorystore (Redis) | 4GB インスタンス | - |
| **分析** | BigQuery | On-demand | - |
| **監視** | Cloud Monitoring | Standard | Cloud Logging |
| **GPU計算資源** | GKE + T4 GPUs | プリエンプティブルノード | Cloud Run (T4) |

### 処理モード

1. **リアルタイム処理**
* アップロード → Cloud Run API → Pub/Sub (優先) → 並列処理 → 判定エンジン → レスポンス (< 2秒)
* **ユースケース:** アップロード時のスクリーニング、コンテンツモデレーション


2. **バッチ処理**
* 一括アップロード → Cloud Storage → Cloud Functions (トリガー) → Pub/Sub (バッチ) → ワーカープール → BigQuery (結果)
* **ユースケース:** 定期監査、データセットのスクリーニング


3. **スケジュールスキャン**
* Cloud Scheduler → Cloud Workflows → データベースクエリ → 比較 → アラート
* **ユースケース:** 既存コンテンツの監視
