# GCP-Based AI-Powered Copyright Infringement Detection System

## System Architecture Overview

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Ingestion Layer                          │
│  • Cloud Storage Upload (Signed URLs)                       │
│  • Cloud Pub/Sub (Message Queue)                            │
│  • Input Validation & Preprocessing                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Extraction Layer                    │
│  ┌──────────────────┐         ┌─────────────────┐          │
│  │ Visual Pipeline  │         │  OCR Pipeline   │          │
│  │  • Perceptual    │         │  • Cloud Vision │          │
│  │    Hashing       │         │    API          │          │
│  │  • Vertex AI     │         │  • Document AI  │          │
│  │    Multimodal    │         │  • Text         │          │
│  │    Embeddings    │         │    Analysis     │          │
│  │  • Feature       │         │                 │          │
│  │    Vectors       │         │                 │          │
│  └──────────────────┘         └─────────────────┘          │
└────────────┬──────────────────────────┬────────────────────┘
             │                          │
             ▼                          ▼
┌─────────────────────────┐  ┌──────────────────────────────┐
│  Similarity Matching    │  │  Copyright Text Analysis     │
│  • Vertex AI Vector     │  │  • Pattern Matching          │
│  •   Search (Matching   │  │  • Entity Recognition        │
│      Engine)            │  │  • Year/Publisher Extraction │
│  • Approximate NN       │  │  • Natural Language API      │
│  • Multi-level Matching │  │                              │
└────────────┬────────────┘  └─────────────┬────────────────┘
             │                              │
             └──────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Decision Engine                           │
│  • Score Aggregation & Weighting                            │
│  • Threshold-based Classification                           │
│  • Explainability Generator                                 │
│  • Cloud Functions (Orchestration)                          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Layer                              │
│  • Match Results (JSON/Protobuf)                            │
│  • Confidence Scores & Evidence                             │
│  • Audit Logs (Cloud Logging)                               │
│  • Analytics (BigQuery)                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Design

### 1. **Ingestion Layer**

**Purpose**: Accept images, validate, preprocess, and route to processing pipelines using GCP-native services.

**Components**:

#### **A. Cloud Storage (Input Bucket)**
- **Configuration**: 
  - Multi-regional bucket for high availability
  - Object lifecycle management (auto-delete processed images after 30 days)
  - Uniform bucket-level access with IAM
  - Object versioning enabled for audit trails

#### **B. Cloud Storage Triggers + Cloud Pub/Sub**
- **Event Flow**:
  ```
  Image Upload → Cloud Storage Trigger → Pub/Sub Message → Cloud Run/Functions
  ```
- **Pub/Sub Topics**:
  - `image-upload-events`: New image notifications
  - `batch-processing-queue`: Bulk processing requests
  - `priority-queue`: High-priority real-time checks
  - `retry-queue`: Failed processing attempts with exponential backoff

#### **C. API Gateway (Cloud Endpoints + Cloud Run)**
- **Endpoints**:
  - `POST /api/v1/check/image` - Single image upload
  - `POST /api/v1/check/batch` - Batch upload (returns job ID)
  - `GET /api/v1/status/{job_id}` - Check processing status
  - `GET /api/v1/results/{job_id}` - Retrieve results

#### **D. Preprocessing Service (Cloud Functions Gen2)**
```python
# Deployed as Cloud Function
import functions_framework
from google.cloud import storage
from PIL import Image
import io

@functions_framework.cloud_event
def preprocess_image(cloud_event):
    """
    Triggered by Cloud Storage upload
    - Validates image format
    - Normalizes dimensions (max 2048px)
    - Converts to standard format (JPEG)
    - Publishes to processing queue
    """
    bucket_name = cloud_event.data["bucket"]
    file_name = cloud_event.data["name"]
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Download and validate
    image_bytes = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Validation
    if image.format not in ['JPEG', 'PNG', 'WEBP']:
        raise ValueError(f"Unsupported format: {image.format}")
    
    # Normalize dimensions (preserve aspect ratio)
    max_dimension = 2048
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Save normalized version
    output_buffer = io.BytesIO()
    image.save(output_buffer, format='JPEG', quality=95)
    
    # Upload to processing bucket
    processed_blob = bucket.blob(f"processed/{file_name}")
    processed_blob.upload_from_string(
        output_buffer.getvalue(),
        content_type='image/jpeg'
    )
    
    # Publish to Pub/Sub for downstream processing
    publish_to_processing_queue({
        'image_path': f"gs://{bucket_name}/processed/{file_name}",
        'original_path': f"gs://{bucket_name}/{file_name}",
        'dimensions': image.size,
        'format': 'JPEG'
    })
```

**Flow**:
```
User Upload → Signed URL → Cloud Storage → Storage Trigger 
→ Preprocessing Function → Validation → Pub/Sub → Processing Pipeline
```

---

### 2. **Visual Similarity Pipeline**

This is the core matching engine using multiple complementary techniques, all GCP-native.

#### **A. Perceptual Hashing (Fast Approximate Matching)**

**Implementation on GCP**:

```python
# Deployed as Cloud Run service
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
    Generates multiple perceptual hashes for robustness
    """
    data = request.json
    image_path = data['image_path']
    
    # Download from Cloud Storage
    storage_client = storage.Client()
    bucket_name, blob_name = parse_gcs_path(image_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    image_bytes = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Generate multiple hashes for redundancy
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
    Query Firestore for similar hashes using Hamming distance
    Threshold: max bit difference (0-256 for 16x16 hash)
    """
    results = []
    
    # Query against reference database
    # Note: Firestore doesn't support Hamming distance natively
    # We'll retrieve candidates and compute in-memory
    
    ref_docs = db.collection('perceptual_hashes').stream()
    
    for doc in ref_docs:
        ref_hashes = doc.to_dict()
        
        # Calculate Hamming distance for each hash type
        distances = {
            'phash': hamming_distance(query_hashes['phash'], ref_hashes['phash']),
            'dhash': hamming_distance(query_hashes['dhash'], ref_hashes['dhash']),
            'whash': hamming_distance(query_hashes['whash'], ref_hashes['whash']),
        }
        
        # If ANY hash is below threshold, consider it a match
        min_distance = min(distances.values())
        
        if min_distance <= threshold:
            results.append({
                'ref_id': doc.id,
                'distances': distances,
                'min_distance': min_distance,
                'match_type': 'perceptual_hash'
            })
    
    # Sort by minimum distance
    results.sort(key=lambda x: x['min_distance'])
    return results[:10]  # Top 10 matches

def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two hashes"""
    return bin(int(hash1, 16) ^ int(hash2, 16)).count('1')
```

**Storage Strategy**:
- **Firestore**: Store hash values with metadata
  - Collection: `perceptual_hashes`
  - Document structure:
    ```json
    {
      "image_id": "copyright_img_12345",
      "phash": "8f373714141c3c3c",
      "dhash": "c3c3c3c3c3c3c3c3",
      "whash": "1e1e1e1e1e1e1e1e",
      "publisher": "PublisherX",
      "copyright_year": 2023,
      "indexed_at": "2024-01-15T10:30:00Z"
    }
    ```

**Optimization for Large Scale**:
- Use **Memorystore (Redis)** for hash lookup cache
- Implement BK-tree structure in Redis for sub-linear Hamming distance queries
- Partition by hash prefix for distributed queries

---

#### **B. Deep Learning Embeddings (Semantic Similarity)**

**Vertex AI Multimodal Embeddings API**:

```python
# Cloud Run service for embedding generation
from google.cloud import aiplatform
from google.cloud import storage
import base64

aiplatform.init(project='your-project-id', location='us-central1')

def generate_multimodal_embedding(image_gcs_path):
    """
    Generate embeddings using Vertex AI Multimodal Embeddings
    Supports: 128, 256, 512, or 1408 dimensions
    """
    
    # Read image from Cloud Storage
    storage_client = storage.Client()
    bucket_name, blob_name = parse_gcs_path(image_gcs_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    image_bytes = blob.download_as_bytes()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Call Vertex AI Multimodal Embeddings API
    from vertexai.vision_models import MultiModalEmbeddingModel
    
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    
    embeddings = model.get_embeddings(
        image=aiplatform.gapic.types.Image(image_bytes=image_bytes),
        dimension=1408  # Highest quality
    )
    
    return {
        'embedding': embeddings.image_embedding,
        'dimension': 1408,
        'model': 'multimodalembedding@001'
    }

# Alternative: Self-hosted CLIP on GKE
def generate_clip_embedding_gke(image_bytes):
    """
    For cost optimization or custom models,
    deploy CLIP on GKE with GPU nodes
    """
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import io
    
    # Load model (should be cached in container)
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    image = Image.open(io.BytesIO(image_bytes))
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # Normalize for cosine similarity
    embedding = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy().flatten().tolist()
```

**Model Selection Matrix**:

| Use Case | Recommended Model | Dimension | Deployment |
|----------|------------------|-----------|------------|
| **Production (Managed)** | Vertex AI Multimodal | 1408 | Fully managed API |
| **Cost-Optimized** | Vertex AI Multimodal | 512 | Managed API |
| **Custom Fine-tuning** | CLIP ViT-L/14 on GKE | 768 | GKE GPU nodes |
| **Ultra Low Latency** | CLIP ViT-B/32 on GKE | 512 | GKE GPU nodes |
| **Highest Accuracy** | DINOv2 ViT-g/14 on GKE | 1536 | GKE GPU nodes |

---

#### **C. Vertex AI Vector Search (Matching Engine)**

**Index Creation & Configuration**:

```python
from google.cloud import aiplatform_v1
from google.cloud import storage
import json

def create_vector_search_index():
    """
    Create Vertex AI Vector Search index for efficient similarity search
    """
    
    client = aiplatform_v1.IndexServiceClient(
        client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    )
    
    # Define index configuration
    index_config = {
        "display_name": "copyright-image-embeddings",
        "description": "Vector search index for copyright detection",
        "metadata": {
            "config": {
                "dimensions": 1408,
                "approximate_neighbors_count": 150,
                "distance_measure_type": "COSINE_DISTANCE",
                "algorithm_config": {
                    "tree_ah_config": {
                        "leaf_node_embedding_count": 500,
                        "leaf_nodes_to_search_percent": 7
                    }
                },
                "shard_size": "SHARD_SIZE_SMALL"  # or MEDIUM, LARGE
            }
        },
        "index_update_method": "STREAM_UPDATE"  # Real-time updates
    }
    
    parent = f"projects/{PROJECT_ID}/locations/us-central1"
    
    operation = client.create_index(parent=parent, index=index_config)
    result = operation.result()  # Wait for completion
    
    return result.name

def deploy_index_endpoint():
    """
    Deploy index to an endpoint for querying
    """
    client = aiplatform_v1.IndexEndpointServiceClient(
        client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    )
    
    endpoint_config = {
        "display_name": "copyright-detection-endpoint",
        "network": f"projects/{PROJECT_ID}/global/networks/default",
        "public_endpoint_enabled": False  # Use VPC for security
    }
    
    parent = f"projects/{PROJECT_ID}/locations/us-central1"
    operation = client.create_index_endpoint(parent=parent, index_endpoint=endpoint_config)
    endpoint = operation.result()
    
    # Deploy index to endpoint
    deployed_index = {
        "id": "copyright_v1",
        "index": INDEX_NAME,
        "dedicated_resources": {
            "machine_spec": {
                "machine_type": "n1-standard-16"  # Scale as needed
            },
            "min_replica_count": 2,
            "max_replica_count": 10
        }
    }
    
    operation = client.deploy_index(
        index_endpoint=endpoint.name,
        deployed_index=deployed_index
    )
    operation.result()
    
    return endpoint.name

def search_similar_images(query_embedding, top_k=10):
    """
    Query Vertex AI Vector Search for similar images
    """
    client = aiplatform_v1.MatchServiceClient(
        client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    )
    
    # Format query
    datapoint = aiplatform_v1.IndexDatapoint(
        feature_vector=query_embedding
    )
    
    query = aiplatform_v1.FindNeighborsRequest.Query(
        datapoint=datapoint,
        neighbor_count=top_k
    )
    
    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=ENDPOINT_NAME,
        deployed_index_id="copyright_v1",
        queries=[query],
        return_full_datapoint=False
    )
    
    response = client.find_neighbors(request)
    
    # Parse results
    matches = []
    for neighbor in response.nearest_neighbors[0].neighbors:
        matches.append({
            'id': neighbor.datapoint.datapoint_id,
            'distance': neighbor.distance,
            'similarity_score': 1 - neighbor.distance  # Convert distance to similarity
        })
    
    return matches
```

**Batch Indexing Pipeline**:

```python
def batch_index_reference_images():
    """
    Batch process reference copyright images and add to Vector Search
    """
    
    # Step 1: List all reference images in Cloud Storage
    storage_client = storage.Client()
    bucket = storage_client.bucket('copyright-reference-images')
    blobs = bucket.list_blobs(prefix='publishers/')
    
    # Step 2: Generate embeddings in batches
    batch_size = 100
    embeddings_batch = []
    
    for i, blob in enumerate(blobs):
        image_bytes = blob.download_as_bytes()
        
        # Generate embedding
        embedding = generate_multimodal_embedding(f"gs://{bucket.name}/{blob.name}")
        
        embeddings_batch.append({
            'id': f"ref_{i}",
            'embedding': embedding['embedding'],
            'metadata': {
                'gcs_path': f"gs://{bucket.name}/{blob.name}",
                'publisher': extract_publisher_from_path(blob.name),
                'indexed_at': datetime.utcnow().isoformat()
            }
        })
        
        # Upload batch when full
        if len(embeddings_batch) >= batch_size:
            upload_to_vector_search(embeddings_batch)
            embeddings_batch = []
    
    # Upload remaining
    if embeddings_batch:
        upload_to_vector_search(embeddings_batch)

def upload_to_vector_search(embeddings_batch):
    """
    Stream update to Vertex AI Vector Search
    """
    client = aiplatform_v1.IndexServiceClient(
        client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
    )
    
    datapoints = []
    for item in embeddings_batch:
        datapoint = aiplatform_v1.IndexDatapoint(
            datapoint_id=item['id'],
            feature_vector=item['embedding']
        )
        datapoints.append(datapoint)
    
    request = aiplatform_v1.UpsertDatapointsRequest(
        index=INDEX_NAME,
        datapoints=datapoints
    )
    
    client.upsert_datapoints(request)
```

**Performance Characteristics**:
- **Query Latency**: 50-100ms for 10M vectors (p95)
- **Throughput**: 1000+ QPS per endpoint
- **Index Update**: Real-time streaming updates (< 1 min propagation)
- **Scalability**: Billions of vectors with automatic sharding

---

### 3. **OCR Pipeline**

**Multi-Engine Approach with GCP Services**:

#### **A. Cloud Vision API (Primary OCR)**

```python
from google.cloud import vision
import re

def detect_copyright_text_vision_api(image_gcs_path):
    """
    Use Cloud Vision API for comprehensive text detection
    """
    client = vision.ImageAnnotatorClient()
    
    image = vision.Image()
    image.source.image_uri = image_gcs_path
    
    # Use DOCUMENT_TEXT_DETECTION for dense text
    response = client.document_text_detection(image=image)
    
    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")
    
    # Extract full text
    full_text = response.full_text_annotation.text
    
    # Also get text with bounding boxes for visualization
    text_annotations = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    
                    # Get bounding box
                    vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                    
                    text_annotations.append({
                        'text': word_text,
                        'confidence': word.confidence,
                        'bounding_box': vertices
                    })
    
    return {
        'full_text': full_text,
        'annotations': text_annotations,
        'language': response.full_text_annotation.pages[0].property.detected_languages[0].language_code if response.full_text_annotation.pages else 'unknown'
    }

def analyze_copyright_patterns(ocr_result):
    """
    Pattern matching for copyright indicators
    """
    text = ocr_result['full_text']
    annotations = ocr_result['annotations']
    
    patterns = {
        # Copyright symbols
        'copyright_symbol': r'[©Ⓒⓒ℗®™]',
        
        # Copyright phrases
        'copyright_word': r'\b[Cc]opyright\b',
        'rights_reserved': r'\b[Aa]ll\s+[Rr]ights\s+[Rr]eserved\b',
        'registered': r'\b[Rr]egistered\s+[Tt]rademark\b',
        
        # Year patterns
        'year_basic': r'\b(19|20)\d{2}\b',
        'year_range': r'\b(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}\b',
        
        # Publisher patterns
        'publisher_with_symbol': r'©\s*(\d{4})?\s*([A-Z][a-zA-Z\s&.,]+(?:Inc|LLC|Ltd|Corp|Publishing|Press|Media)?)',
        'publisher_with_word': r'Copyright\s*©?\s*(\d{4})?\s*(?:by\s+)?([A-Z][a-zA-Z\s&.,]+)',
        
        # Attribution patterns
        'photo_credit': r'(?:Photo|Image)\s+(?:by|credit|©)\s*:?\s*([A-Z][a-zA-Z\s]+)',
        'photographer': r'Photograph(?:er|y)?\s+(?:by\s+)?([A-Z][a-zA-Z\s]+)',
    }
    
    matches = {}
    bounding_boxes = {}
    
    for key, pattern in patterns.items():
        found = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        matches[key] = [match.group() for match in found]
        
        # Find bounding boxes for matches (approximate)
        if matches[key]:
            bounding_boxes[key] = find_text_bounding_boxes(
                matches[key], 
                annotations
            )
    
    # Calculate confidence scores
    confidence_score = calculate_copyright_confidence(matches)
    
    return {
        'raw_text': text,
        'copyright_indicators': matches,
        'bounding_boxes': bounding_boxes,
        'confidence_score': confidence_score,
        'detected_language': ocr_result.get('language', 'unknown'),
        'metadata': extract_copyright_metadata(matches)
    }

def calculate_copyright_confidence(matches):
    """
    Calculate OCR-based copyright detection confidence
    """
    score = 0
    weights = {
        'copyright_symbol': 40,
        'copyright_word': 30,
        'rights_reserved': 20,
        'registered': 15,
        'year_basic': 10,
        'year_range': 15,
        'publisher_with_symbol': 35,
        'publisher_with_word': 30,
        'photo_credit': 25,
        'photographer': 20
    }
    
    for key, weight in weights.items():
        if matches.get(key):
            score += weight
    
    # Normalize to 0-100
    return min(score, 100)

def extract_copyright_metadata(matches):
    """
    Extract structured copyright information
    """
    metadata = {
        'copyright_years': [],
        'publishers': [],
        'credits': []
    }
    
    # Extract years
    if matches.get('year_basic'):
        metadata['copyright_years'] = list(set(matches['year_basic']))
    
    # Extract publishers
    if matches.get('publisher_with_symbol'):
        for match in matches['publisher_with_symbol']:
            # Parse publisher name (simplified)
            metadata['publishers'].append(match)
    
    # Extract credits
    if matches.get('photo_credit'):
        metadata['credits'].extend(matches['photo_credit'])
    
    return metadata

def find_text_bounding_boxes(search_texts, annotations):
    """
    Find bounding boxes for specific text matches
    """
    boxes = []
    
    for search_text in search_texts:
        search_words = search_text.lower().split()
        
        for i, annotation in enumerate(annotations):
            if annotation['text'].lower() in search_words:
                # Find consecutive words that match
                matching_annotations = [annotation]
                j = i + 1
                matched_words = 1
                
                while j < len(annotations) and matched_words < len(search_words):
                    if annotations[j]['text'].lower() in search_words:
                        matching_annotations.append(annotations[j])
                        matched_words += 1
                    j += 1
                
                if matched_words == len(search_words):
                    # Combine bounding boxes
                    combined_box = combine_bounding_boxes(
                        [a['bounding_box'] for a in matching_annotations]
                    )
                    boxes.append({
                        'text': search_text,
                        'box': combined_box,
                        'confidence': sum(a['confidence'] for a in matching_annotations) / len(matching_annotations)
                    })
                    break
    
    return boxes

def combine_bounding_boxes(boxes):
    """
    Combine multiple bounding boxes into one
    """
    all_x = []
    all_y = []
    
    for box in boxes:
        for vertex in box:
            all_x.append(vertex[0])
            all_y.append(vertex[1])
    
    return [
        (min(all_x), min(all_y)),  # Top-left
        (max(all_x), min(all_y)),  # Top-right
        (max(all_x), max(all_y)),  # Bottom-right
        (min(all_x), max(all_y))   # Bottom-left
    ]
```

#### **B. Document AI (For Complex Documents)**

```python
from google.cloud import documentai_v1 as documentai

def process_complex_document(image_gcs_path, processor_id):
    """
    Use Document AI for complex copyright notices in PDFs/multi-page documents
    """
    client = documentai.DocumentProcessorServiceClient()
    
    # Read document from Cloud Storage
    storage_client = storage.Client()
    bucket_name, blob_name = parse_gcs_path(image_gcs_path)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    document_bytes = blob.download_as_bytes()
    
    # Configure processor
    name = client.processor_path(PROJECT_ID, LOCATION, processor_id)
    
    # Prepare document
    raw_document = documentai.RawDocument(
        content=document_bytes,
        mime_type='application/pdf'  # or 'image/jpeg', 'image/png'
    )
    
    request = documentai.ProcessRequest(
        name=name,
        raw_document=raw_document
    )
    
    result = client.process_document(request=request)
    document = result.document
    
    # Extract entities (Document AI can be trained to recognize copyright entities)
    copyright_entities = []
    for entity in document.entities:
        if entity.type_ in ['copyright_year', 'publisher', 'rights_statement']:
            copyright_entities.append({
                'type': entity.type_,
                'text': entity.mention_text,
                'confidence': entity.confidence,
                'page': entity.page_anchor.page_refs[0].page if entity.page_anchor.page_refs else 0
            })
    
    return {
        'full_text': document.text,
        'entities': copyright_entities,
        'pages': len(document.pages)
    }
```

#### **C. Natural Language API (Entity & Sentiment Analysis)**

```python
from google.cloud import language_v1

def analyze_copyright_context(text):
    """
    Use Natural Language API to understand context and extract entities
    """
    client = language_v1.LanguageServiceClient()
    
    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT
    )
    
    # Entity analysis
    entities_response = client.analyze_entities(
        request={'document': document}
    )
    
    # Extract relevant entities
    organizations = []
    persons = []
    dates = []
    
    for entity in entities_response.entities:
        if entity.type_ == language_v1.Entity.Type.ORGANIZATION:
            organizations.append({
                'name': entity.name,
                'salience': entity.salience,
                'mentions': len(entity.mentions)
            })
        elif entity.type_ == language_v1.Entity.Type.PERSON:
            persons.append({
                'name': entity.name,
                'salience': entity.salience
            })
        elif entity.type_ == language_v1.Entity.Type.DATE:
            dates.append(entity.name)
    
    return {
        'organizations': sorted(organizations, key=lambda x: x['salience'], reverse=True),
        'persons': sorted(persons, key=lambda x: x['salience'], reverse=True),
        'dates': dates
    }
```

**OCR Pipeline Orchestration** (Cloud Workflows):

```yaml
# copyright_ocr_workflow.yaml
main:
  params: [image_path]
  steps:
    - ocr_vision:
        call: googleapis.vision.v1.images.annotate
        args:
          body:
            requests:
              - image:
                  source:
                    imageUri: ${image_path}
                features:
                  - type: DOCUMENT_TEXT_DETECTION
        result: vision_result
    
    - analyze_patterns:
        call: http.post
        args:
          url: https://CLOUD_RUN_URL/analyze_copyright
          body:
            ocr_result: ${vision_result}
        result: pattern_result
    
    - entity_extraction:
        call: googleapis.language.v1.documents.analyzeEntities
        args:
          body:
            document:
              type: PLAIN_TEXT
              content: ${vision_result.responses[0].fullTextAnnotation.text}
        result: entity_result
    
    - combine_results:
        return:
          ocr_data: ${vision_result}
          copyright_patterns: ${pattern_result}
          entities: ${entity_result}
```

---

### 4. **Decision Engine**

**Comprehensive Scoring System** (Cloud Functions):

```python
import functions_framework
from google.cloud import firestore
import json

db = firestore.Client()

@functions_framework.http
def calculate_copyright_score(request):
    """
    Main decision engine that combines all analysis results
    """
    data = request.get_json()
    
    visual_results = data['visual_results']
    ocr_results = data['ocr_results']
    image_metadata = data.get('metadata', {})
    
    # 1. Visual Similarity Score (0-100)
    visual_score = calculate_visual_score(visual_results)
    
    # 2. OCR Copyright Indicators Score (0-100)
    ocr_score = calculate_ocr_score(ocr_results)
    
    # 3. Metadata Alignment Score (0-100)
    metadata_score = calculate_metadata_score(
        visual_results, 
        ocr_results, 
        image_metadata
    )
    
    # 4. Weighted Final Score
    weights = {
        'visual': 0.60,      # Visual match is primary signal
        'ocr': 0.30,         # Copyright text is strong indicator
        'metadata': 0.10     # Metadata consistency check
    }
    
    final_score = (
        visual_score * weights['visual'] +
        ocr_score * weights['ocr'] +
        metadata_score * weights['metadata']
    )
    
    # 5. Risk Classification
    classification = classify_risk(final_score, visual_results, ocr_results)
    
    # 6. Generate Explanation
    explanation = generate_explanation(
        final_score,
        visual_score,
        ocr_score,
        metadata_score,
        visual_results,
        ocr_results
    )
    
    # 7. Compile Evidence
    evidence = compile_evidence(visual_results, ocr_results)
    
    result = {
        'overall_score': round(final_score, 2),
        'classification': classification,
        'component_scores': {
            'visual_similarity': round(visual_score, 2),
            'copyright_text': round(ocr_score, 2),
            'metadata_alignment': round(metadata_score, 2)
        },
        'risk_level': classification['level'],
        'confidence': classification['confidence'],
        'explanation': explanation,
        'evidence': evidence,
        'timestamp': firestore.SERVER_TIMESTAMP
    }
    
    # Store result in Firestore
    store_result(image_metadata.get('job_id'), result)
    
    return json.dumps(result)

def calculate_visual_score(visual_results):
    """
    Score based on image similarity metrics
    """
    score = 0
    
    # Check vector search results
    if visual_results.get('vector_matches'):
        top_match = visual_results['vector_matches'][0]
        similarity = top_match['similarity_score']
        
        if similarity >= 0.98:
            score = 100
        elif similarity >= 0.95:
            score = 95
        elif similarity >= 0.90:
            score = 85
        elif similarity >= 0.85:
            score = 75
        elif similarity >= 0.80:
            score = 65
        elif similarity >= 0.75:
            score = 50
        elif similarity >= 0.70:
            score = 35
        else:
            score = max(0, (similarity - 0.5) * 100)
    
    # Boost score if perceptual hash also matches
    if visual_results.get('hash_matches'):
        hash_match = visual_results['hash_matches'][0]
        hamming_distance = hash_match['min_distance']
        
        if hamming_distance <= 5:
            score = max(score, 90)
        elif hamming_distance <= 10:
            score = max(score, 75)
        elif hamming_distance <= 15:
            score = max(score, 60)
    
    # Multiple strong matches increase confidence
    if len(visual_results.get('vector_matches', [])) >= 3:
        avg_top3_similarity = sum(
            m['similarity_score'] for m in visual_results['vector_matches'][:3]
        ) / 3
        if avg_top3_similarity >= 0.85:
            score = min(100, score + 10)
    
    return min(100, score)

def calculate_ocr_score(ocr_results):
    """
    Score based on copyright text detection
    """
    if not ocr_results or not ocr_results.get('copyright_indicators'):
        return 0
    
    indicators = ocr_results['copyright_indicators']
    score = 0
    
    # Scoring weights for each indicator
    weights = {
        'copyright_symbol': 35,
        'copyright_word': 25,
        'rights_reserved': 20,
        'registered': 15,
        'year_basic': 8,
        'year_range': 12,
        'publisher_with_symbol': 30,
        'publisher_with_word': 25,
        'photo_credit': 20,
        'photographer': 15
    }
    
    for indicator, weight in weights.items():
        if indicators.get(indicator):
            # Give partial credit based on number of matches
            count = len(indicators[indicator])
            score += min(weight, weight * (0.5 + 0.5 * min(count, 3) / 3))
    
    # Bonus for complete copyright notice
    has_symbol = bool(indicators.get('copyright_symbol'))
    has_year = bool(indicators.get('year_basic') or indicators.get('year_range'))
    has_owner = bool(indicators.get('publisher_with_symbol') or indicators.get('publisher_with_word'))
    
    if has_symbol and has_year and has_owner:
        score += 15  # Bonus for complete notice
    
    return min(100, score)

def calculate_metadata_score(visual_results, ocr_results, metadata):
    """
    Cross-validate information consistency
    """
    score = 50  # Baseline neutral score
    
    if not visual_results.get('vector_matches'):
        return score
    
    top_match = visual_results['vector_matches'][0]
    
    # Check if OCR-detected publisher matches visual match metadata
    if ocr_results.get('metadata', {}).get('publishers'):
        detected_publisher = ocr_results['metadata']['publishers'][0]
        reference_publisher = top_match.get('metadata', {}).get('publisher', '')
        
        if detected_publisher.lower() in reference_publisher.lower():
            score += 30  # Strong alignment
        elif similar_text(detected_publisher, reference_publisher) > 0.7:
            score += 15  # Partial alignment
    
    # Check year consistency
    if ocr_results.get('metadata', {}).get('copyright_years'):
        detected_year = int(ocr_results['metadata']['copyright_years'][0])
        reference_year = top_match.get('metadata', {}).get('copyright_year')
        
        if reference_year and abs(detected_year - reference_year) <= 1:
            score += 20
    
    return min(100, score)

def classify_risk(final_score, visual_results, ocr_results):
    """
    Classify copyright infringement risk level
    """
    # High confidence thresholds
    if final_score >= 90:
        level = "CRITICAL"
        confidence = "very_high"
        action = "immediate_review_required"
    elif final_score >= 80:
        level = "HIGH"
        confidence = "high"
        action = "manual_review_recommended"
    elif final_score >= 65:
        level = "MEDIUM"
        confidence = "medium"
        action = "review_if_public_facing"
    elif final_score >= 45:
        level = "LOW"
        confidence = "low"
        action = "monitor"
    else:
        level = "MINIMAL"
        confidence = "very_low"
        action = "no_action_needed"
    
    # Adjust confidence based on agreement between signals
    visual_score = calculate_visual_score(visual_results)
    ocr_score = calculate_ocr_score(ocr_results)
    
    # If both high, boost confidence
    if visual_score >= 85 and ocr_score >= 70:
        confidence = "very_high"
    # If conflict (one high, one low), reduce confidence
    elif abs(visual_score - ocr_score) > 40:
        confidence = lower_confidence(confidence)
    
    return {
        'level': level,
        'confidence': confidence,
        'recommended_action': action,
        'requires_human_review': final_score >= 65
    }

def generate_explanation(final_score, visual_score, ocr_score, metadata_score, 
                        visual_results, ocr_results):
    """
    Generate human-readable explanation of the decision
    """
    explanation_parts = []
    
    # Visual matching explanation
    if visual_results.get('vector_matches'):
        top_match = visual_results['vector_matches'][0]
        similarity_pct = round(top_match['similarity_score'] * 100, 1)
        
        if similarity_pct >= 95:
            explanation_parts.append(
                f"Image shows {similarity_pct}% visual similarity to copyrighted "
                f"reference image '{top_match['id']}'"
            )
        elif similarity_pct >= 75:
            explanation_parts.append(
                f"Moderate visual similarity ({similarity_pct}%) detected with "
                f"reference image '{top_match['id']}'"
            )
    
    # Hash matching explanation
    if visual_results.get('hash_matches'):
        hash_match = visual_results['hash_matches'][0]
        if hash_match['min_distance'] <= 10:
            explanation_parts.append(
                f"Perceptual hash analysis indicates near-duplicate or modified version "
                f"(Hamming distance: {hash_match['min_distance']})"
            )
    
    # Copyright text explanation
    if ocr_results.get('copyright_indicators'):
        indicators = ocr_results['copyright_indicators']
        found_indicators = []
        
        if indicators.get('copyright_symbol'):
            found_indicators.append("copyright symbol (©)")
        if indicators.get('copyright_word'):
            found_indicators.append("'Copyright' text")
        if indicators.get('rights_reserved'):
            found_indicators.append("'All Rights Reserved' notice")
        
        if found_indicators:
            explanation_parts.append(
                f"Copyright notice detected containing: {', '.join(found_indicators)}"
            )
        
        # Publisher information
        if ocr_results.get('metadata', {}).get('publishers'):
            publisher = ocr_results['metadata']['publishers'][0]
            explanation_parts.append(
                f"Publisher identified: {publisher}"
            )
    
    # Overall assessment
    if final_score >= 85:
        explanation_parts.insert(0, 
            "Strong evidence of copyrighted material. Multiple signals confirm match."
        )
    elif final_score >= 65:
        explanation_parts.insert(0,
            "Moderate copyright risk detected. Review recommended."
        )
    else:
        explanation_parts.insert(0,
            "Low copyright risk. Visual or text similarity below threshold."
        )
    
    return " | ".join(explanation_parts)

def compile_evidence(visual_results, ocr_results):
    """
    Compile all evidence for audit trail
    """
    evidence = {
        'visual_matches': [],
        'copyright_text': {},
        'bounding_boxes': []
    }
    
    # Visual match evidence
    if visual_results.get('vector_matches'):
        for match in visual_results['vector_matches'][:5]:
            evidence['visual_matches'].append({
                'reference_id': match['id'],
                'similarity_score': round(match['similarity_score'], 4),
                'reference_metadata': match.get('metadata', {}),
                'thumbnail_url': f"gs://copyright-references/thumbnails/{match['id']}.jpg"
            })
    
    # Copyright text evidence
    if ocr_results.get('raw_text'):
        evidence['copyright_text'] = {
            'full_text': ocr_results['raw_text'][:500],  # Truncate for storage
            'indicators': ocr_results.get('copyright_indicators', {}),
            'metadata': ocr_results.get('metadata', {})
        }
    
    # Bounding box evidence for visualization
    if ocr_results.get('bounding_boxes'):
        evidence['bounding_boxes'] = ocr_results['bounding_boxes']
    
    return evidence

def store_result(job_id, result):
    """
    Store analysis result in Firestore
    """
    doc_ref = db.collection('copyright_analysis_results').document(job_id)
    doc_ref.set(result)

def lower_confidence(current_confidence):
    """Helper to reduce confidence level"""
    levels = ['very_low', 'low', 'medium', 'high', 'very_high']
    current_idx = levels.index(current_confidence)
    return levels[max(0, current_idx - 1)]

def similar_text(text1, text2):
    """Calculate text similarity (simple Jaccard similarity)"""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0
```

---

### 5. **Orchestration & Workflow**

**Cloud Workflows Definition**:

```yaml
# complete_copyright_check_workflow.yaml
main:
  params: [input]
  steps:
    - init:
        assign:
          - job_id: ${input.job_id}
          - image_path: ${input.image_path}
          - mode: ${input.mode}  # "realtime" or "batch"
    
    - parallel_processing:
        parallel:
          branches:
            - visual_pipeline:
                steps:
                  - generate_hashes:
                      call: http.post
                      args:
                        url: ${HASH_SERVICE_URL}
                        body:
                          image_path: ${image_path}
                      result: hash_result
                  
                  - generate_embedding:
                      call: http.post
                      args:
                        url: ${EMBEDDING_SERVICE_URL}
                        body:
                          image_path: ${image_path}
                      result: embedding_result
                  
                  - search_similar:
                      call: http.post
                      args:
                        url: ${VECTOR_SEARCH_URL}
                        body:
                          embedding: ${embedding_result.embedding}
                          top_k: 10
                      result: visual_matches
            
            - ocr_pipeline:
                steps:
                  - ocr_detection:
                      call: googleapis.vision.v1.images.annotate
                      args:
                        body:
                          requests:
                            - image:
                                source:
                                  imageUri: ${image_path}
                              features:
                                - type: DOCUMENT_TEXT_DETECTION
                      result: ocr_raw
                  
                  - analyze_copyright:
                      call: http.post
                      args:
                        url: ${COPYRIGHT_ANALYSIS_URL}
                        body:
                          ocr_result: ${ocr_raw}
                      result: ocr_analyzed
    
    - decision_engine:
        call: http.post
        args:
          url: ${DECISION_ENGINE_URL}
          body:
            visual_results:
              vector_matches: ${visual_matches}
              hash_matches: ${hash_result}
            ocr_results: ${ocr_analyzed}
            metadata:
              job_id: ${job_id}
              image_path: ${image_path}
        result: final_decision
    
    - store_and_notify:
        steps:
          - store_result:
              call: googleapis.firestore.v1.projects.databases.documents.patch
              args:
                name: ${"projects/" + PROJECT_ID + "/databases/(default)/documents/results/" + job_id}
                body: ${final_decision}
          
          - conditional_notification:
              switch:
                - condition: ${final_decision.classification.level == "CRITICAL"}
                  steps:
                    - send_alert:
                        call: http.post
                        args:
                          url: ${ALERT_WEBHOOK_URL}
                          body:
                            severity: "CRITICAL"
                            job_id: ${job_id}
                            score: ${final_decision.overall_score}
    
    - return_result:
        return: ${final_decision}
```

---

## Cloud Deployment Architecture

### **GCP Infrastructure Stack**

```
┌──────────────────────────────────────────────────────────────┐
│                    Cloud Load Balancing                       │
│                  (Global HTTPS Load Balancer)                 │
└──────────────┬──────────────────────────────────────────────┬┘
               │                                                 │
               ▼                                                 ▼
┌──────────────────────────┐                     ┌────────────────────────┐
│   Cloud Run (API)        │                     │  Cloud Run (Workers)   │
│  • FastAPI Service       │                     │  • Embedding Generator │
│  • Auto-scaling 0-1000   │                     │  • Hash Processor      │
│  • CPU: 4 vCPU           │                     │  • GPU: T4 (optional)  │
│  • Memory: 8GB           │                     │  • Auto-scale 0-100    │
└────────┬─────────────────┘                     └───────────┬────────────┘
         │                                                    │
         ▼                                                    ▼
┌──────────────────────────────────────────────────────────────┐
│                      Cloud Pub/Sub                            │
│  Topics:                                                      │
│    • image-uploads (main queue)                              │
│    • priority-checks (real-time)                             │
│    • batch-processing (bulk jobs)                            │
│    • results-notifications                                   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                   Processing Layer                            │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Cloud Functions│  │ GKE Workloads│  │  Cloud Workflows │  │
│  │  • Preprocessor│  │  • GPU Nodes │  │  • Orchestration │  │
│  │  • Validator   │  │  • CLIP Model│  │  • Error Handling│  │
│  └────────────────┘  └──────────────┘  └─────────────────┘  │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                    GCP AI/ML Services                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Vertex AI                                             │   │
│  │  • Multimodal Embeddings API                         │   │
│  │  • Vector Search (Matching Engine)                   │   │
│  │  • Prediction Endpoints                              │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Cloud Vision API                                      │   │
│  │  • Document Text Detection                           │   │
│  │  • Logo Detection (optional)                         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Document AI (optional)                                │   │
│  │  • Complex Document Processing                       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                    Data Storage Layer                         │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Cloud Storage  │  │  Firestore   │  │  Memorystore    │  │
│  │  • Input       │  │  • Metadata  │  │  (Redis)        │  │
│  │  • Reference   │  │  • Results   │  │  • Hash Cache   │  │
│  │  • Thumbnails  │  │  • Jobs      │  │  • Session Data │  │
│  └────────────────┘  └──────────────┘  └─────────────────┘  │
│  ┌────────────────┐  ┌──────────────┐                        │
│  │  BigQuery      │  │  Cloud SQL   │                        │
│  │  • Analytics   │  │  • Audit Log │                        │
│  │  • Reporting   │  │  • User Data │                        │
│  └────────────────┘  └──────────────┘                        │
└───────────────────────────────────────────────────────────────┘
```

### **Detailed Service Configuration**

#### **1. Cloud Run Services**

```yaml
# api-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: copyright-api
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "1000"
        autoscaling.knative.dev/target: "100"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
      - image: gcr.io/PROJECT_ID/copyright-api:latest
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
        env:
        - name: PROJECT_ID
          value: "your-project-id"
        - name: PUBSUB_TOPIC
          value: "image-uploads"
```

```yaml
# worker-service.yaml (GPU-enabled)
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: embedding-worker
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/gpu-type: "nvidia-tesla-t4"
    spec:
      containers:
      - image: gcr.io/PROJECT_ID/embedding-worker:latest
        resources:
          limits:
            cpu: "4"
            memory: "16Gi"
            nvidia.com/gpu: "1"
```

#### **2. GKE Cluster (For Custom Models)**

```yaml
# gke-gpu-nodepool.yaml
apiVersion: container.cnrm.cloud.google.com/v1beta1
kind: ContainerNodePool
metadata:
  name: gpu-nodepool
spec:
  clusterRef:
    name: copyright-detection-cluster
  initialNodeCount: 0
  autoscaling:
    minNodeCount: 0
    maxNodeCount: 20
  nodeConfig:
    machineType: n1-standard-4
    guestAccelerator:
    - type: nvidia-tesla-t4
      count: 1
    preemptible: true  # Cost optimization
```

```yaml
# clip-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: clip-embedding-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: clip-embeddings
  template:
    metadata:
      labels:
        app: clip-embeddings
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
      - name: clip-service
        image: gcr.io/PROJECT_ID/clip-service:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        ports:
        - containerPort: 8080
```

#### **3. Vertex AI Vector Search Configuration**

```python
# Setup script: setup_vertex_vector_search.py
from google.cloud import aiplatform

def setup_production_index():
    """
    Production-grade Vector Search configuration
    """
    
    # Initialize Vertex AI
    aiplatform.init(
        project='your-project-id',
        location='us-central1'
    )
    
    # Create index with optimal settings
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name="copyright-embeddings-prod",
        dimensions=1408,
        approximate_neighbors_count=150,
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=10,
        distance_measure_type="COSINE_DISTANCE",
        description="Production copyright detection index",
        labels={"env": "production", "version": "v1"}
    )
    
    # Create endpoint
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name="copyright-endpoint-prod",
        public_endpoint_enabled=False,
        description="Production endpoint for copyright detection"
    )
    
    # Deploy index to endpoint
    endpoint.deploy_index(
        index=index,
        deployed_index_id="copyright_prod_v1",
        machine_type="n1-standard-16",
        min_replica_count=2,
        max_replica_count=20,
        enable_access_logging=True
    )
    
    print(f"Index: {index.resource_name}")
    print(f"Endpoint: {endpoint.resource_name}")
    
    return index, endpoint
```

---

## Technology Stack Summary

| Component | GCP Service | Configuration | Alternatives |
|-----------|-------------|---------------|--------------|
| **API Gateway** | Cloud Run | 4 vCPU, 8GB RAM | Cloud Functions |
| **Message Queue** | Cloud Pub/Sub | Standard tier | - |
| **Image Storage** | Cloud Storage | Multi-regional | - |
| **Perceptual Hashing** | Cloud Run | Custom Python | Cloud Functions |
| **Deep Learning** | Vertex AI Embeddings | 1408-dim | GKE + CLIP |
| **Vector Database** | Vertex AI Vector Search | COSINE, ANN | Self-hosted Milvus on GKE |
| **OCR Engine** | Cloud Vision API | DOCUMENT_TEXT | Document AI |
| **Text Analysis** | Natural Language API | Entity extraction | - |
| **Orchestration** | Cloud Workflows | YAML definition | Cloud Composer |
| **Metadata DB** | Firestore | Native mode | Cloud SQL PostgreSQL |
| **Caching** | Memorystore (Redis) | 4GB instance | - |
| **Analytics** | BigQuery | On-demand | - |
| **Monitoring** | Cloud Monitoring | Standard | Cloud Logging |
| **GPU Compute** | GKE + T4 GPUs | Preemptible nodes | Cloud Run (T4) |

---

## Processing Modes

### **1. Real-Time Processing**
```
Upload → Cloud Run API → Pub/Sub (priority) → Parallel Processing 
→ Decision Engine → Response < 2s
```

**Configuration**:
- Min instances: 2 (warm pool)
- Timeout: 10 seconds
- Use case: Upload screening, content moderation

### **2. Batch Processing**
```
Bulk Upload → Cloud Storage → Cloud Functions (trigger) → Pub/Sub (batch) 
→ Worker Pool → BigQuery (results)
```

**Configuration**:
- Workers: Auto-scale 0-100
- Throughput: 50,000 images/hour
- Use case: Periodic audits, dataset screening

### **3. Scheduled Scanning**
```
Cloud Scheduler → Cloud Workflows → Query Database → Compare → Alert
```

**Configuration**:
- Schedule: Daily at 2 AM UTC
- Use case: Monitor existing content

---
