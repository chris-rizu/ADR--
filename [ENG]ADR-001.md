# AI-Powered Copyright Infringement Detection System

## System Architecture Overview

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Ingestion Layer                          │
│  • Image Upload API (REST/gRPC)                             │
│  • Batch Processing Queue (SQS/Kafka)                       │
│  • Input Validation & Preprocessing                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Extraction Layer                    │
│  ┌──────────────────┐         ┌─────────────────┐          │
│  │ Visual Pipeline  │         │  OCR Pipeline   │          │
│  │  • Perceptual    │         │  • Tesseract    │          │
│  │    Hashing       │         │  • EasyOCR      │          │
│  │  • CLIP/DINO     │         │  • PaddleOCR    │          │
│  │    Embeddings    │         │  • Text         │          │
│  │  • Feature       │         │    Extraction   │          │
│  │    Vectors       │         │                 │          │
│  └──────────────────┘         └─────────────────┘          │
└────────────┬──────────────────────────┬────────────────────┘
             │                          │
             ▼                          ▼
┌─────────────────────────┐  ┌──────────────────────────────┐
│  Similarity Matching    │  │  Copyright Text Analysis     │
│  • Vector DB Search     │  │  • Pattern Matching          │
│  • Approximate NN       │  │  • Entity Recognition        │
│  • Multi-level Matching │  │  • Year/Publisher Extraction │
└────────────┬────────────┘  └─────────────┬────────────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Decision Engine                           │
│  • Score Aggregation & Weighting                            │
│  • Threshold-based Classification                           │
│  • Explainability Generator                                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Layer                              │
│  • Match Results (JSON/Protobuf)                            │
│  • Confidence Scores & Evidence                             │
│  • Audit Logs & Analytics                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Design

### 1. **Ingestion Layer**

**Purpose**: Accept images, validate, preprocess, and route to processing pipelines.

**Components**:
- **API Gateway**: FastAPI or AWS API Gateway for REST endpoints
- **Message Queue**: Apache Kafka or AWS SQS for batch processing
- **Preprocessing**: Image normalization (resize to standard dimensions, format conversion)

**Flow**:
```
User Upload → API Validation → Deduplicate → Queue → Processing
```

---

### 2. **Visual Similarity Pipeline**

This is the core matching engine using multiple complementary techniques.

#### **A. Perceptual Hashing (Fast Approximate Matching)**

**Algorithms**:
- **pHash** (Perceptual Hash): Robust to minor modifications
- **dHash** (Difference Hash): Fast, good for exact/near-duplicates
- **wHash** (Wavelet Hash): Better for cropping resistance

**Use Case**: Quick first-pass filtering for exact/near-exact matches

**Implementation**:
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

**Storage**: Redis or specialized hash databases with Hamming distance queries

---

#### **B. Deep Learning Embeddings (Semantic Similarity)**

**Recommended Models**:

1. **CLIP (Contrastive Language-Image Pre-training)**
   - **Model**: OpenAI CLIP (ViT-L/14 or ViT-B/32)
   - **Strengths**: Multimodal understanding, robust to transformations
   - **Embedding Size**: 512 or 768 dimensions
   - **Use**: Primary semantic similarity

2. **DINOv2** (Self-supervised Vision Transformer)
   - **Model**: Meta's DINOv2 (ViT-L/14 or ViT-g/14)
   - **Strengths**: Excellent visual features without text, better for fine-grained matching
   - **Use**: Secondary matching for visual-heavy content

3. **EfficientNet or ResNet** (as fallback)
   - **Model**: EfficientNet-B7 or ResNet-152
   - **Use**: Lightweight alternative for resource-constrained scenarios

**Implementation Example**:
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

#### **C. Vector Database for Efficient Retrieval**

**Recommended Solutions**:

1. **Pinecone** (Managed, easiest)
   - Auto-scaling, low latency
   - Built-in HNSW indexing
   - Good for production

2. **Milvus** (Open-source, flexible)
   - Self-hosted or Zilliz Cloud
   - Supports GPU acceleration
   - Excellent for large-scale deployments

3. **FAISS** (Facebook AI Similarity Search)
   - Library-level integration
   - Requires custom infrastructure
   - Best performance if self-managed

4. **Weaviate** or **Qdrant** (Alternatives)
   - Good balance of features and ease

**Indexing Strategy**:
- Use **HNSW** (Hierarchical Navigable Small World) for approximate nearest neighbor search
- Partition by image category/publisher for faster queries
- Maintain separate indices for different embedding models

**Query Flow**:
```
Input Image → Extract Embedding → Vector DB Query (k-NN) 
→ Return top-N candidates with similarity scores
```

**Scalability**:
- **Millions of images**: Single Milvus/Pinecone cluster
- **Hundreds of millions**: Sharded indices by category/date
- **Sub-100ms latency** achievable with proper GPU acceleration

---

### 3. **OCR Pipeline**

**Multi-Engine Approach** for robustness:

1. **Tesseract OCR** (Open-source baseline)
   - Version 5.x with LSTM models
   - Good for clean, horizontal text

2. **EasyOCR** (Deep learning-based)
   - Handles multiple orientations
   - Better with complex backgrounds
   - 80+ language support

3. **PaddleOCR** (Advanced)
   - Superior for rotated/distorted text
   - Lightweight and fast

4. **Cloud OCR APIs** (Optional fallback)
   - Google Cloud Vision API
   - AWS Textract
   - Azure Computer Vision

**Implementation**:
```python
import easyocr
import re

reader = easyocr.Reader(['en'])

def extract_copyright_text(image_path):
    results = reader.readtext(image_path)
    text = ' '.join([item[1] for item in results])
    
    # Pattern matching for copyright symbols
    patterns = {
        'copyright_symbol': r'[©Ⓒⓒ]',
        'copyright_word': r'\b[Cc]opyright\b',
        'rights_reserved': r'[Aa]ll [Rr]ights [Rr]eserved',
        'year': r'\b(19|20)\d{2}\b',
        'publisher': r'(?:©|Copyright)\s*(\d{4})?\s*([A-Z][a-zA-Z\s&]+)',
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

**Preprocessing for OCR**:
- Grayscale conversion
- Contrast enhancement (CLAHE)
- Deskewing/rotation correction
- Noise reduction

---

### 4. **Decision Engine**

**Scoring Algorithm**:

```python
def calculate_copyright_score(visual_results, ocr_results):
    # Visual similarity component (0-100)
    visual_score = 0
    
    if visual_results['top_match_similarity'] > 0.95:
        visual_score = 100
    elif visual_results['top_match_similarity'] > 0.85:
        visual_score = 80
    elif visual_results['top_match_similarity'] > 0.70:
        visual_score = 50
    elif visual_results['perceptual_hash_distance'] < 5:
        visual_score = 70
    
    # OCR component (0-100)
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
    
    # Weighted combination
    final_score = (visual_score * 0.70) + (ocr_score * 0.30)
    
    # Confidence classification
    if final_score >= 85:
        classification = "HIGH_RISK"
    elif final_score >= 60:
        classification = "MEDIUM_RISK"
    elif final_score >= 40:
        classification = "LOW_RISK"
    else:
        classification = "MINIMAL_RISK"
    
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

**Explainability Features**:
- Return matched reference image IDs and thumbnails
- Highlight detected copyright text with bounding boxes
- Show similarity heatmaps (where visual match occurs)
- Provide reasoning (e.g., "Matched due to 94% visual similarity with Image ID 12345 + detected © symbol")

---

## Cloud Deployment Architecture

### **Recommended Stack**

**Infrastructure**: AWS/GCP/Azure (example with AWS)

```
┌─────────────────────────────────────────────────────────┐
│                     CloudFront CDN                       │
│                  (API Distribution)                      │
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
│  (API Service)   │    │ (Worker Service) │
│  • FastAPI       │    │  • Image Proc    │
│  • Async Queue   │    │  • GPU Instances │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│   Amazon SQS     │    │   S3 Buckets     │
│  (Job Queue)     │    │  • Input Images  │
└──────────────────┘    │  • Reference DB  │
                        └──────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│          Processing Infrastructure           │
│  ┌────────────┐  ┌────────────┐            │
│  │  ECS GPU   │  │  Lambda    │            │
│  │  (CLIP)    │  │  (OCR)     │            │
│  └────────────┘  └────────────┘            │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│           Data Storage Layer                 │
│  • Pinecone/Milvus (Vector DB)              │
│  • ElastiCache Redis (Hash Storage)         │
│  • RDS PostgreSQL (Metadata)                │
│  • DynamoDB (Results Cache)                 │
└─────────────────────────────────────────────┘
```

### **Scaling Considerations**

**Compute**:
- **API Layer**: Auto-scaling ECS tasks (CPU-based scaling)
- **GPU Workers**: g4dn.xlarge or p3.2xlarge instances for embedding extraction
- **OCR Workers**: Lambda (for light loads) or CPU-heavy ECS tasks

**Storage**:
- **Reference Images**: S3 Standard (with CloudFront for distribution)
- **Vector Embeddings**: Pinecone (managed scaling) or Milvus cluster
- **Metadata**: RDS PostgreSQL with read replicas

**Performance Targets**:
- **Real-time**: < 2 seconds for single image (with warm cache)
- **Batch**: 10,000+ images/hour per worker node
- **Throughput**: Scale to millions of images/day with horizontal scaling

---

## Processing Modes

### **1. Real-Time Processing**
```
Upload → Immediate Processing → Response in < 3s
```
- Use case: Content moderation, upload screening
- Architecture: Synchronous API + warm GPU instances

### **2. Batch Processing**
```
Bulk Upload → Queue → Parallel Workers → Aggregated Report
```
- Use case: Periodic audits, large dataset screening
- Architecture: SQS + Auto-scaling worker fleet

### **3. Scheduled Scanning**
```
Crawl Database → Compare Against Updated Reference Library
```
- Use case: Monitoring existing content against new copyrighted material
- Architecture: Cron jobs + incremental indexing

---

## Technology Stack Summary

| Component | Recommended Technology | Alternatives |
|-----------|------------------------|--------------|
| **API Framework** | FastAPI (Python) | Flask, Express.js |
| **Image Processing** | Pillow, OpenCV | scikit-image |
| **Perceptual Hashing** | ImageHash library | Custom implementation |
| **Deep Learning** | PyTorch + Transformers | TensorFlow |
| **Embedding Model** | CLIP ViT-L/14 | DINOv2, ResNet |
| **Vector Database** | Pinecone, Milvus | FAISS, Weaviate |
| **OCR Engine** | EasyOCR, PaddleOCR | Tesseract, Cloud APIs |
| **Message Queue** | AWS SQS, Kafka | RabbitMQ, Redis Streams |
| **Object Storage** | AWS S3 | GCS, Azure Blob |
| **Metadata DB** | PostgreSQL | MongoDB, DynamoDB |
| **Caching** | Redis | Memcached |
| **Orchestration** | Kubernetes, ECS | Docker Swarm |
| **Monitoring** | Prometheus + Grafana | CloudWatch, Datadog |

---

## Implementation Roadmap

**Phase 1: MVP (4-6 weeks)**
- Set up ingestion API
- Implement perceptual hashing pipeline
- Basic OCR integration (Tesseract)
- Simple scoring algorithm
- PostgreSQL metadata storage

**Phase 2: Enhanced Matching (6-8 weeks)**
- Integrate CLIP embeddings
- Deploy vector database (Pinecone/Milvus)
- Multi-OCR engine support
- Improved decision engine with ML-based scoring

**Phase 3: Production Scaling (8-10 weeks)**
- Kubernetes deployment
- GPU acceleration
- Batch processing pipeline
- Monitoring and alerting
- A/B testing framework for threshold tuning

**Phase 4: Advanced Features (Ongoing)**
- Active learning to improve matching
- Publisher-specific customization
- Blockchain-based proof of detection
- Integration with DMCA takedown workflows

---

## Cost Optimization Strategies

1. **Use spot instances** for batch GPU workloads (60-70% savings)
2. **Cache embeddings** for reference images (compute once, store forever)
3. **Tiered matching**: Fast perceptual hash first, expensive deep learning only for uncertain cases
4. **Lazy OCR**: Only run OCR if visual match is ambiguous
5. **Smart image preprocessing**: Downscale images for embedding extraction (512x512 often sufficient)

---

## Existing Services & Frameworks to Consider

**Commercial Solutions**:
- **Google Vision API**: Reverse image search + OCR (expensive at scale)
- **Amazon Rekognition**: Custom labels + text detection
- **Microsoft Azure Computer Vision**: Similar capabilities

**Open-Source Frameworks**:
- **TinEye**: Commercial reverse image search (API available)
- **PhotoDNA** (Microsoft): Primarily for child safety content
- **Content Authenticity Initiative (CAI)**: Standards for content provenance

**Hybrid Approach**: Use commercial APIs for validation/spot-checking while running custom system for primary processing to control costs and IP.

---

## Key Performance Metrics

Monitor these to ensure system health:

- **Precision**: % of flagged images that are actual infringements
- **Recall**: % of actual infringements detected
- **Latency**: p50, p95, p99 processing times
- **Throughput**: Images processed per hour
- **False Positive Rate**: Critical for user trust
- **Vector DB Query Time**: Should be < 100ms for real-time
- **OCR Accuracy**: Character/word-level accuracy

---

This architecture provides a production-ready foundation that's modular, scalable, and explainable. You can start with the MVP components and incrementally add sophistication as your dataset and requirements grow.
