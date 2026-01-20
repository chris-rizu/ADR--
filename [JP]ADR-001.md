# AI駆動型著作権侵害検出システム

## システムアーキテクチャ概要

大規模な著作権コンテンツ検出のための、本番環境対応の包括的なアーキテクチャを設計します。

---

## ハイレベルアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                     取り込みレイヤー                          │
│  • 画像アップロードAPI (REST/gRPC)                           │
│  • バッチ処理キュー (SQS/Kafka)                              │
│  • 入力検証・前処理                                          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  特徴抽出レイヤー                             │
│  ┌──────────────────┐         ┌─────────────────┐          │
│  │ 視覚パイプライン  │         │  OCRパイプライン │          │
│  │  • パーセプチュアル│         │  • Tesseract    │          │
│  │    ハッシュ       │         │  • EasyOCR      │          │
│  │  • CLIP/DINO     │         │  • PaddleOCR    │          │
│  │    埋め込み      │         │  • テキスト抽出  │          │
│  │  • 特徴ベクトル   │         │                 │          │
│  └──────────────────┘         └─────────────────┘          │
└────────────┬──────────────────────────┬────────────────────┘
             │                          │
             ▼                          ▼
┌─────────────────────────┐  ┌──────────────────────────────┐
│  類似度マッチング        │  │  著作権テキスト分析           │
│  • ベクトルDB検索       │  │  • パターンマッチング         │
│  • 近似最近傍探索       │  │  • エンティティ認識           │
│  • 多層マッチング       │  │  • 年号・出版社抽出           │
└────────────┬────────────┘  └─────────────┬────────────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    判定エンジン                              │
│  • スコア集約・重み付け                                      │
│  • 閾値ベース分類                                           │
│  • 説明可能性生成                                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    出力レイヤー                              │
│  • マッチング結果 (JSON/Protobuf)                           │
│  • 信頼度スコア・証拠                                        │
│  • 監査ログ・分析                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 詳細コンポーネント設計

### 1. **取り込みレイヤー**

**目的**: 画像を受け取り、検証、前処理を行い、処理パイプラインへルーティングする。

**コンポーネント**:
- **APIゲートウェイ**: FastAPIまたはAWS API Gateway（RESTエンドポイント用）
- **メッセージキュー**: Apache KafkaまたはAWS SQS（バッチ処理用）
- **前処理**: 画像正規化（標準寸法へのリサイズ、フォーマット変換）

**フロー**:
```
ユーザーアップロード → API検証 → 重複排除 → キュー → 処理
```

---

### 2. **視覚類似度パイプライン**

複数の補完的な技術を使用したコアマッチングエンジンです。

#### **A. パーセプチュアルハッシュ（高速近似マッチング）**

**アルゴリズム**:
- **pHash**（パーセプチュアルハッシュ）: 軽微な変更に強い
- **dHash**（差分ハッシュ）: 高速、完全一致・準一致に適している
- **wHash**（ウェーブレットハッシュ）: トリミング耐性が高い

**使用例**: 完全一致・準一致の高速一次フィルタリング

**実装**:
```python
from imagehash import phash, dhash, whash
from PIL import Image

def generate_perceptual_hashes(image_path):
    img = Image.open(image_path)
    return {
        'phash': str(phash(img)),
        'dhash': str(dhash(img)),
        'whash': str(whash(img))
    }
```

**ストレージ**: Redisまたはハミング距離クエリに対応した専用ハッシュデータベース

---

#### **B. ディープラーニング埋め込み（意味的類似性）**

**推奨モデル**:

1. **CLIP（Contrastive Language-Image Pre-training）**
   - **モデル**: OpenAI CLIP（ViT-L/14またはViT-B/32）
   - **強み**: マルチモーダル理解、変換に強い
   - **埋め込みサイズ**: 512または768次元
   - **用途**: 主要な意味的類似性検出

2. **DINOv2**（自己教師あり Vision Transformer）
   - **モデル**: Meta の DINOv2（ViT-L/14またはViT-g/14）
   - **強み**: テキストなしで優れた視覚特徴、細粒度マッチングに最適
   - **用途**: 視覚重視コンテンツの二次マッチング

3. **EfficientNetまたはResNet**（フォールバック）
   - **モデル**: EfficientNet-B7またはResNet-152
   - **用途**: リソース制約のあるシナリオ向けの軽量代替案

**実装例**:
```python
import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def extract_clip_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.cpu().numpy().flatten()
```

---

#### **C. 効率的な検索のためのベクトルデータベース**

**推奨ソリューション**:

1. **Pinecone**（マネージド、最も簡単）
   - 自動スケーリング、低レイテンシ
   - 組み込みHNSWインデックス
   - 本番環境に最適

2. **Milvus**（オープンソース、柔軟性が高い）
   - セルフホストまたはZilliz Cloud
   - GPUアクセラレーションサポート
   - 大規模展開に最適

3. **FAISS**（Facebook AI Similarity Search）
   - ライブラリレベルの統合
   - カスタムインフラストラクチャが必要
   - セルフマネージドの場合は最高のパフォーマンス

4. **WeaviateまたはQdrant**（代替案）
   - 機能と使いやすさのバランスが良い

**インデックス戦略**:
- 近似最近傍探索には**HNSW**（Hierarchical Navigable Small World）を使用
- より高速なクエリのために画像カテゴリ/出版社ごとにパーティション分割
- 異なる埋め込みモデル用に別々のインデックスを維持

**クエリフロー**:
```
入力画像 → 埋め込み抽出 → ベクトルDBクエリ（k-NN）
→ 類似度スコア付きトップN候補を返す
```

**スケーラビリティ**:
- **数百万枚の画像**: 単一のMilvus/Pineconeクラスター
- **数億枚の画像**: カテゴリ/日付別にシャード化されたインデックス
- 適切なGPUアクセラレーションで**100ms以下のレイテンシ**を達成可能

---

### 3. **OCRパイプライン**

堅牢性のための**マルチエンジンアプローチ**:

1. **Tesseract OCR**（オープンソースベースライン）
   - LSTMモデルを搭載したバージョン5.x
   - クリーンで水平なテキストに適している

2. **EasyOCR**（ディープラーニングベース）
   - 複数の向きに対応
   - 複雑な背景に強い
   - 80以上の言語サポート

3. **PaddleOCR**（高度）
   - 回転/歪んだテキストに優れている
   - 軽量で高速

4. **クラウドOCR API**（オプションのフォールバック）
   - Google Cloud Vision API
   - AWS Textract
   - Azure Computer Vision

**実装**:
```python
import easyocr
import re

reader = easyocr.Reader(['ja', 'en'])  # 日本語と英語をサポート

def extract_copyright_text(image_path):
    results = reader.readtext(image_path)
    text = ' '.join([item[1] for item in results])
    
    # 著作権記号のパターンマッチング
    patterns = {
        'copyright_symbol': r'[©Ⓒⓒ]',
        'copyright_word': r'\b[Cc]opyright|著作権\b',
        'rights_reserved': r'[Aa]ll [Rr]ights [Rr]eserved|無断転載禁止|無断複製禁止',
        'year': r'\b(19|20)\d{2}\b|[令平昭]\d{1,2}年',
        'publisher': r'(?:©|Copyright|著作権)\s*(\d{4}年?)?(\d{4})?\s*([ぁ-んァ-ヶー一-龠A-Z][ぁ-んァ-ヶー一-龠a-zA-Z\s&株式会社]+)',
    }
    
    matches = {}
    for key, pattern in patterns.items():
        matches[key] = re.findall(pattern, text)
    
    return {
        'raw_text': text,
        'copyright_indicators': matches,
        'confidence': calculate_ocr_confidence(results)
    }
```

**OCR用前処理**:
- グレースケール変換
- コントラスト強調（CLAHE）
- 傾き補正/回転補正
- ノイズ除去

---

### 4. **判定エンジン**

**スコアリングアルゴリズム**:

```python
def calculate_copyright_score(visual_results, ocr_results):
    # 視覚類似度コンポーネント（0-100）
    visual_score = 0
    
    if visual_results['top_match_similarity'] > 0.95:
        visual_score = 100
    elif visual_results['top_match_similarity'] > 0.85:
        visual_score = 80
    elif visual_results['top_match_similarity'] > 0.70:
        visual_score = 50
    elif visual_results['perceptual_hash_distance'] < 5:
        visual_score = 70
    
    # OCRコンポーネント（0-100）
    ocr_score = 0
    indicators = ocr_results['copyright_indicators']
    
    if indicators['copyright_symbol']:
        ocr_score += 40
    if indicators['copyright_word']:
        ocr_score += 30
    if indicators['rights_reserved']:
        ocr_score += 20
    if indicators['publisher']:
        ocr_score += 30
    
    ocr_score = min(ocr_score, 100)
    
    # 重み付け組み合わせ
    final_score = (visual_score * 0.70) + (ocr_score * 0.30)
    
    # 信頼度分類
    if final_score >= 85:
        classification = "高リスク"
    elif final_score >= 60:
        classification = "中リスク"
    elif final_score >= 40:
        classification = "低リスク"
    else:
        classification = "最小リスク"
    
    return {
        'overall_score': final_score,
        'classification': classification,
        'visual_score': visual_score,
        'ocr_score': ocr_score,
        'evidence': {
            'matched_images': visual_results['top_matches'],
            'copyright_text': ocr_results['raw_text'],
            'detected_indicators': indicators
        }
    }
```

**説明可能性機能**:
- マッチした参照画像IDとサムネイルを返す
- 検出された著作権テキストをバウンディングボックスでハイライト
- 類似度ヒートマップを表示（視覚的マッチが発生した場所）
- 理由を提供（例:「画像ID 12345と94%の視覚的類似性でマッチ + ©記号検出」）

---

## クラウド展開アーキテクチャ

### **推奨スタック**

**インフラストラクチャ**: AWS/GCP/Azure（AWSの例）

```
┌─────────────────────────────────────────────────────────┐
│                     CloudFront CDN                       │
│                  (API配信)                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Application Load Balancer                   │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│   ECS Fargate    │    │   ECS Fargate    │
│  (APIサービス)    │    │ (ワーカーサービス)│
│  • FastAPI       │    │  • 画像処理      │
│  • 非同期キュー   │    │  • GPUインスタンス│
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│   Amazon SQS     │    │   S3バケット     │
│  (ジョブキュー)   │    │  • 入力画像      │
└──────────────────┘    │  • 参照DB        │
                        └──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│          処理インフラストラクチャ             │
│  ┌────────────┐  ┌────────────┐            │
│  │  ECS GPU   │  │  Lambda    │            │
│  │  (CLIP)    │  │  (OCR)     │            │
│  └────────────┘  └────────────┘            │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│           データストレージレイヤー            │
│  • Pinecone/Milvus（ベクトルDB）            │
│  • ElastiCache Redis（ハッシュストレージ）  │
│  • RDS PostgreSQL（メタデータ）             │
│  • DynamoDB（結果キャッシュ）               │
└─────────────────────────────────────────────┘
```

### **スケーリング考慮事項**

**コンピュート**:
- **APIレイヤー**: 自動スケーリングECSタスク（CPUベーススケーリング）
- **GPUワーカー**: g4dn.xlargeまたはp3.2xlargeインスタンス（埋め込み抽出用）
- **OCRワーカー**: Lambda（軽負荷用）またはCPU集約型ECSタスク

**ストレージ**:
- **参照画像**: S3 Standard（CloudFront配信付き）
- **ベクトル埋め込み**: Pinecone（マネージドスケーリング）またはMilvusクラスター
- **メタデータ**: 読み取りレプリカ付きRDS PostgreSQL

**パフォーマンス目標**:
- **リアルタイム**: 単一画像で2秒未満（キャッシュがウォーム状態）
- **バッチ**: ワーカーノードあたり10,000枚以上の画像/時間
- **スループット**: 水平スケーリングで1日あたり数百万枚の画像に対応

---

## 処理モード

### **1. リアルタイム処理**
```
アップロード → 即時処理 → 3秒以内にレスポンス
```
- 使用例: コンテンツモデレーション、アップロードスクリーニング
- アーキテクチャ: 同期API + ウォームGPUインスタンス

### **2. バッチ処理**
```
一括アップロード → キュー → 並列ワーカー → 集約レポート
```
- 使用例: 定期監査、大規模データセットスクリーニング
- アーキテクチャ: SQS + 自動スケーリングワーカーフリート

### **3. スケジュールスキャン**
```
データベースクロール → 更新された参照ライブラリと比較
```
- 使用例: 新しい著作権素材に対する既存コンテンツの監視
- アーキテクチャ: Cronジョブ + 増分インデックス

---

## 技術スタック概要

| コンポーネント | 推奨技術 | 代替案 |
|-----------|---------|--------|
| **APIフレームワーク** | FastAPI (Python) | Flask, Express.js |
| **画像処理** | Pillow, OpenCV | scikit-image |
| **パーセプチュアルハッシュ** | ImageHashライブラリ | カスタム実装 |
| **ディープラーニング** | PyTorch + Transformers | TensorFlow |
| **埋め込みモデル** | CLIP ViT-L/14 | DINOv2, ResNet |
| **ベクトルDB** | Pinecone, Milvus | FAISS, Weaviate |
| **OCRエンジン** | EasyOCR, PaddleOCR | Tesseract, クラウドAPI |
| **メッセージキュー** | AWS SQS, Kafka | RabbitMQ, Redis Streams |
| **オブジェクトストレージ** | AWS S3 | GCS, Azure Blob |
| **メタデータDB** | PostgreSQL | MongoDB, DynamoDB |
| **キャッシング** | Redis | Memcached |
| **オーケストレーション** | Kubernetes, ECS | Docker Swarm |
| **監視** | Prometheus + Grafana | CloudWatch, Datadog |

---

## 実装ロードマップ

**フェーズ1: MVP（4-6週間）**
- 取り込みAPIのセットアップ
- パーセプチュアルハッシュパイプラインの実装
- 基本的なOCR統合（Tesseract）
- シンプルなスコアリングアルゴリズム
- PostgreSQLメタデータストレージ

**フェーズ2: 強化されたマッチング（6-8週間）**
- CLIP埋め込みの統合
- ベクトルデータベースのデプロイ（Pinecone/Milvus）
- マルチOCRエンジンサポート
- MLベーススコアリングによる改善された判定エンジン

**フェーズ3: 本番スケーリング（8-10週間）**
- Kubernetesデプロイメント
- GPUアクセラレーション
- バッチ処理パイプライン
- 監視とアラート
- 閾値調整のためのA/Bテストフレームワーク

**フェーズ4: 高度な機能（継続的）**
- マッチングを改善するためのアクティブラーニング
- 出版社固有のカスタマイズ
- ブロックチェーンベースの検出証明
- DMCA削除ワークフローとの統合

---

## コスト最適化戦略

1. バッチGPUワークロード用に**スポットインスタンス**を使用（60-70%のコスト削減）
2. 参照画像の**埋め込みをキャッシュ**（一度計算、永続的保存）
3. **段階的マッチング**: 最初に高速パーセプチュアルハッシュ、不確実なケースのみ高コストのディープラーニング
4. **遅延OCR**: 視覚的マッチが曖昧な場合のみOCRを実行
5. **スマート画像前処理**: 埋め込み抽出用に画像をダウンスケール（512x512で十分な場合が多い）

---

## 検討すべき既存サービス・フレームワーク

**商用ソリューション**:
- **Google Vision API**: 逆画像検索 + OCR（大規模では高価）
- **Amazon Rekognition**: カスタムラベル + テキスト検出
- **Microsoft Azure Computer Vision**: 同様の機能

**オープンソースフレームワーク**:
- **TinEye**: 商用逆画像検索（API利用可能）
- **PhotoDNA**（Microsoft）: 主に児童安全コンテンツ用
- **Content Authenticity Initiative (CAI)**: コンテンツ来歴の標準

**ハイブリッドアプローチ**: コストと知的財産を管理するため、一次処理にはカスタムシステムを使用し、検証/スポットチェックには商用APIを使用

---

## 主要パフォーマンス指標

システムの健全性を確保するために、以下を監視します:

- **精度（Precision）**: フラグが立てられた画像のうち、実際の侵害である割合
- **再現率（Recall）**: 実際の侵害のうち検出された割合
- **レイテンシ**: p50、p95、p99処理時間
- **スループット**: 1時間あたりに処理された画像数
- **偽陽性率**: ユーザー信頼性にとって重要
- **ベクトルDBクエリ時間**: リアルタイム処理には100ms未満であるべき
- **OCR精度**: 文字/単語レベルの精度

---

このアーキテクチャは、モジュール式、スケーラブル、説明可能な本番環境対応の基盤を提供します。MVPコンポーネントから始めて、データセットと要件の増加に応じて段階的に高度化を追加できます。
