# EX-IA Indiana / IU-Xray

Pipeline para trabajar exclusivamente con Indiana / IU-Xray en una fase de alineamiento debil imagen-atencion-concepto.

Esta fase se centra en:

- preparar el dataset Indiana / IU-Xray;
- extraer conceptos clinicos debiles desde informes;
- generar un texto sintetico corto de anomalias con Qwen de forma offline;
- entrenar DenseNet121 como extractor visual;
- anadir un MLP residual ligero para clasificacion multilabel;
- generar mapas Grad-CAM;
- entrenar una rama ligera de atencion con decoder/deconvolucion;
- obtener metricas de clasificacion y utilidad de heatmaps.

Por ahora no se ejecuta generacion de informes completa, QA ni alineamiento contrastivo con Qwen. Qwen se puede usar opcionalmente solo para crear una frase sintetica corta por informe.

## 1. Instalar dependencias

Desde la raiz del repo:

```bash
python -m pip install -r requirements.txt
```

Si el server tiene GPU CUDA, instala PyTorch con CUDA antes o sustituye la version CPU por la version CUDA adecuada segun la web oficial de PyTorch.

## 2. Preparar Indiana / IU-Xray completo

OpenI directo puede devolver HTML de error en algunos entornos. Por eso el flujo recomendado usa el mirror Hugging Face `ykumards/open-i`, que contiene los estudios IU-Xray en Parquet con imagenes frontal/lateral e informes.

```bash
python -m src.indiana_xray.prepare_indiana \
  --source hf \
  --data-root data/indiana \
  --hf-image-size 384 \
  --hf-image-format jpg \
  --out-tsv data/indiana/indiana_master.tsv
```

Salidas esperadas:

```text
data/indiana/indiana_master.tsv
data/indiana/images/*.jpg
data/indiana/hf_parquet/*.parquet
```

El TSV maestro incluye rutas de imagen, secciones del informe, etiquetas disponibles, proyeccion y campos de grounding si existieran.

## 3. Extraer conceptos clinicos

```bash
python -m src.indiana_xray.extract_concepts \
  --dataset-tsv data/indiana/indiana_master.tsv \
  --out-tsv data/indiana/indiana_concepts.tsv
```

Salida:

```text
data/indiana/indiana_concepts.tsv
```

Este archivo anade conceptos normalizados por imagen y columnas multilabel `label_*` para entrenamiento visual.

## 4. Generar texto sintetico corto con Qwen

Este paso es opcional. Sirve para crear una frase breve y densa con las anomalias relevantes de cada informe. Esa frase puede usarse mas adelante como texto para CLIP loss y next-token loss.

El prompt fuerza una salida de maximo 15 palabras, solo con hallazgos radiograficos visibles y localizaciones. No debe incluir recomendaciones, follow-up, correlacion clinica, fechas, comparaciones ni metadatos.

El modelo recomendado para esta destilacion textual offline es `Qwen/Qwen2.5-1.5B-Instruct`, porque debe seguir una instruccion de resumen clinico. El alineamiento posterior puede seguir usando `Qwen/Qwen2.5-1.5B` congelado si se quiere mantener la arquitectura original.

Prueba pequena:

```bash
python -m src.indiana_xray.generate_synthetic_text \
  --input-tsv data/indiana/indiana_concepts.tsv \
  --out-tsv data/indiana/indiana_synthetic_debug.tsv \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --limit 20 \
  --batch-size 4 \
  --device cuda \
  --dtype float16
```

Ejecucion completa:

```bash
python -m src.indiana_xray.generate_synthetic_text \
  --input-tsv data/indiana/indiana_concepts.tsv \
  --out-tsv data/indiana/indiana_synthetic.tsv \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 4 \
  --device cuda \
  --dtype float16 \
  --resume
```

Columnas nuevas:

```text
report_text_clean
synthetic_anomaly_text
clip_text
next_token_text
synthetic_model
synthetic_prompt_version
```

Ejemplo de salida esperada:

```text
synthetic_anomaly_text: Mild cardiomegaly with small left pleural effusion.
clip_text: Chest xray: Mild cardiomegaly with small left pleural effusion.
next_token_text: Mild cardiomegaly with small left pleural effusion.
```

Si ya existe un TSV generado con una version anterior del prompt, vuelve a generarlo sin `--resume` o usa otro nombre de salida, por ejemplo:

```bash
python -m src.indiana_xray.generate_synthetic_text \
  --input-tsv data/indiana/indiana_concepts.tsv \
  --out-tsv data/indiana/indiana_synthetic_v2.tsv \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 8 \
  --device cuda \
  --dtype float16
```

## 5. Entrenar DenseNet121 + MLP residual

En GPU:

```bash
python -m src.indiana_xray.train_densenet \
  --concepts-tsv data/indiana/indiana_concepts.tsv \
  --out-dir runs/densenet_full \
  --epochs 5 \
  --batch-size 32 \
  --device cuda
```

Si aparece OOM, baja `--batch-size`:

```bash
--batch-size 16
```

o:

```bash
--batch-size 8
```

Salidas:

```text
runs/densenet_full/best.pt
runs/densenet_full/metrics.json
```

Metricas principales:

- `macro_auc`
- `macro_ap`
- `micro_f1`
- `macro_f1`
- AUC/AP por clase

## 6. Generar Grad-CAM + decoder de atencion

```bash
python -m src.indiana_xray.generate_gradcam \
  --concepts-tsv data/indiana/indiana_concepts.tsv \
  --checkpoint runs/densenet_full/best.pt \
  --out-dir runs/gradcam_full \
  --decoder-epochs 3 \
  --batch-size 32 \
  --device cuda
```

Si aparece OOM, baja `--batch-size` a `16` u `8`.

Salidas:

```text
runs/gradcam_full/gradcam_embeddings.npz
runs/gradcam_full/attention_decoder.pt
runs/gradcam_full/metrics.json
```

Metricas principales:

- `decoder_bce_last`: ajuste del decoder ligero frente al mapa Grad-CAM.
- `mean_deletion_drop_top20`: caida media de probabilidad al ocultar el 20% mas caliente del heatmap.
- `median_deletion_drop_top20`: mediana de esa caida.
- `mean_refined_attention_mse`: distancia media entre atencion refinada y Grad-CAM.

## 7. Ver resultados

```bash
cat runs/densenet_full/metrics.json
cat runs/gradcam_full/metrics.json
```

En Windows PowerShell:

```powershell
Get-Content runs\densenet_full\metrics.json
Get-Content runs\gradcam_full\metrics.json
```

## Nota sobre Qwen

`Qwen/Qwen2.5-1.5B` y `Qwen/Qwen2.5-1.5B-Instruct` pesan varios GB. En este README, Qwen solo se usa de forma opcional para generar `indiana_synthetic.tsv` offline.

Cuando se retome la fase contrastiva, la idea sera mantener Qwen congelado inicialmente, sin anadir tokens. El texto de entrenamiento podra venir de `clip_text` y `next_token_text`.
