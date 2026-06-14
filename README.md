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

## 2b. Preparar MIMIC-CXR-JPG

MIMIC-CXR necesita estar descargado localmente. Este cargador asume la estructura habitual de MIMIC-CXR-JPG:

```text
MIMIC_JPG_ROOT/
  files/p10/p10000032/s50414267/*.jpg
  mimic-cxr-2.0.0-metadata.csv
  mimic-cxr-2.0.0-split.csv
  mimic-cxr-2.0.0-chexpert.csv
```

Y, si tienes los informes de texto:

```text
MIMIC_REPORTS_ROOT/
  files/p10/p10000032/s50414267.txt
```

Comando compatible con el resto del pipeline:

```bash
python -m src.indiana_xray.prepare_mimic_cxr \
  --mimic-jpg-root /path/to/mimic-cxr-jpg/2.0.0 \
  --mimic-reports-root /path/to/mimic-cxr/2.0.0 \
  --out-tsv data/mimic/mimic_master.tsv \
  --synthetic-out-tsv data/mimic/mimic_synthetic.tsv \
  --views PA AP \
  --split all
```

Salidas:

```text
data/mimic/mimic_master.tsv
data/mimic/mimic_synthetic.tsv
```

`mimic_master.tsv` mantiene el mismo esquema basico que Indiana: `image_path`, `image_id`, `projection`, `findings`, `impression`, `report_text`, `labels_available` y columnas `label_*`.

`mimic_synthetic.tsv` usa directamente `IMPRESSION` como texto corto para `clip_text` y `next_token_text`. En MIMIC suele ser mejor que regenerar texto con Qwen, porque la impresion ya es una sintesis radiologica humana.

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

El prompt fuerza una salida de maximo 12 palabras, solo con hallazgos radiograficos visibles y localizaciones. Usa el informe como fuente principal, los conceptos debiles como pistas y una lista precomputada de 20-30 keywords clinicas del propio informe, sin articulos ni stopwords. No debe incluir recomendaciones, follow-up, correlacion clinica, fechas, comparaciones, metadatos, tubos, cateteres, lineas, clips, puertos, marcapasos ni hardware salvo que sean el hallazgo principal.

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

Si se quiere usar un generador mas potente con menos VRAM, se puede cargar Qwen 7B en 4-bit:

```bash
python -m src.indiana_xray.generate_synthetic_text \
  --input-tsv data/indiana/indiana_concepts.tsv \
  --out-tsv data/indiana/indiana_synthetic_v4_qwen7b.tsv \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --batch-size 1 \
  --device cuda \
  --dtype float16 \
  --load-in-4bit \
  --keyword-count 30
```

Columnas nuevas:

```text
report_text_clean
synthetic_anomaly_text
clip_text
next_token_text
prompt_keywords
synthetic_model
synthetic_prompt_version
```

Alternativa cerrada para ablation, util si quieres medir el techo de convergencia con vocabulario fijo y sin generacion abierta:

```bash
python -m src.indiana_xray.make_controlled_synthetic_text \
  --input-tsv data/indiana/indiana_concepts.tsv \
  --out-tsv data/indiana/indiana_synthetic_controlled.tsv \
  --summary-out data/indiana/controlled_terms_summary.tsv \
  --top-k 10 \
  --max-terms-per-image 3
```

Esta version no usa generacion abierta. Primero cuenta los hallazgos mas comunes en `label_*`/`concepts`, mantiene un vocabulario cerrado y genera frases canonicas cortas como `Cardiomegaly and pleural effusion.`. Sirve como comparativa, pero el flujo principal recomendado es Qwen guiado por `prompt_keywords`.

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
  --out-tsv data/indiana/indiana_synthetic_v4.tsv \
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
  --epochs 10 \
  --batch-size 8 \
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
  --decoder-epochs 10 \
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

## 7. Entrenar Qwen para explicabilidad

Este paso usa `indiana_synthetic_v4.tsv`, DenseNet y Grad-CAM para aprender una conexion explicativa visual-textual. Qwen queda congelado: no se anaden tokens y no se genera texto durante el entrenamiento.

Se entrenan solo:

- proyector visual para CLIP loss;
- proyector de prefijo visual para next-token loss;
- temperatura contrastiva.

```bash
python -m src.indiana_xray.train_qwen_explainability \
  --synthetic-tsv data/indiana/indiana_synthetic_v4_qwen7b.tsv \
  --densenet-checkpoint runs/densenet_full_e10/best.pt \
  --gradcam-dir runs/gradcam_full_e10 \
  --out-dir runs/qwen_explainability_e10 \
  --model-id Qwen/Qwen2.5-1.5B \
  --epochs 20 \
  --batch-size 2 \
  --device cuda \
  --dtype float16 \
  --clip-weight 0.5 \
  --next-token-weight 0.5 \
  --prefix-len 8 \
  --eval-examples 6 \
  --eval-random-examples
```

Si hay OOM, baja `--batch-size` a `1`. Si el next-token pesa demasiado frente a CLIP, baja `--next-token-weight` a `0.1`.

Salidas:

```text
runs/qwen_explainability_e10/best.pt
runs/qwen_explainability_e10/metrics.json
```

Metricas principales:

- `retrieval_r1`
- `retrieval_r5`
- `median_rank`
- `positive_cosine`
- `next_token_loss`

## 8. Ver resultados

```bash
cat runs/densenet_full/metrics.json
cat runs/gradcam_full/metrics.json
cat runs/qwen_explainability_e10/metrics.json
```

En Windows PowerShell:

```powershell
Get-Content runs\densenet_full\metrics.json
Get-Content runs\gradcam_full\metrics.json
Get-Content runs\qwen_explainability_e10\metrics.json
```

## Nota sobre Qwen

`Qwen/Qwen2.5-1.5B` y `Qwen/Qwen2.5-1.5B-Instruct` pesan varios GB. En este README, `Qwen/Qwen2.5-1.5B-Instruct` se usa para generar `indiana_synthetic.tsv` offline y `Qwen/Qwen2.5-1.5B` se usa congelado para la fase de explicabilidad.

La fase de explicabilidad mantiene Qwen congelado inicialmente, sin anadir tokens. El texto de entrenamiento viene de `clip_text` y `next_token_text`.
