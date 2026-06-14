import os, json, torch, torch.nn as nn, numpy as np, random, xgboost as xgb_lib
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import math
from peft import LoraConfig, get_peft_model
from pathlib import Path
import argparse
import re
import random
from pathlib import Path
import torch.nn.functional as F

from .models import *
from .models import _maybe_make_4bit_config

    
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--FUZZY_DATA_OUT",
        type=str,
        default="data/toon_mhealth",
        help="Ruta base al dataset fuzzy (ej: data/toon_pamap)"
    )
    return p.parse_args()
    
# --- ENTRENAMIENTO ---
def train():

    args = parse_args()
    data_dir = args.FUZZY_DATA_OUT

    device = "cuda"
    model_id = "Qwen/Qwen2.5-1.5B"
    #model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id,
          "EOS:", tokenizer.eos_token, tokenizer.eos_token_id)
          
    tokenizer.padding_side = "right"
    
    
    print("EOS token:", tokenizer.eos_token)
    print("EOS id:", tokenizer.eos_token_id)
    print("endoftext id:", tokenizer.convert_tokens_to_ids("<|endoftext|>"))

    qconfig, compute_dtype = _maybe_make_4bit_config("4bit")

    llama = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=qconfig,
        torch_dtype=compute_dtype,
        device_map="auto"
    )

    llama.config.use_cache = False  # importante para entreno
    llama.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # reduce VRAM


    #ds_tr = IMUMultimodalDataset("data/toon_mhealth/features_train_flat.npz", "data/toon_mhealth/features_train_zsym.npz", "data/toon_mhealth/xgb_model.json")
    #ds_te = IMUMultimodalDataset("data/toon_mhealth/features_test_flat.npz", "data/toon_mhealth/features_test_zsym.npz", "data/toon_mhealth/xgb_model.json", split="test")
    
    ds_tr = IMUMultimodalDataset(
        f"{data_dir}/features_train_flat.npz",
        f"{data_dir}/features_train_zsym.npz",
        f"{data_dir}/xgb_model.json"
    )

    ds_te = IMUMultimodalDataset(
        f"{data_dir}/features_test_flat.npz",
        f"{data_dir}/features_test_zsym.npz",
        f"{data_dir}/xgb_model.json",
        split="test"
    )

    print(f"📦 Dataset cargado desde: {data_dir}")
    print(f"   - #samples train: {len(ds_tr)}")
    print(f"   - #classes: {len(ds_tr.class_names)}")
    print(f"   - #sensors: {len(ds_tr.sensors)}")

    # --- vocab expand ---
    # --- vocab expand ---
    emb_layer = llama.get_input_embeddings()
    old_vocab = emb_layer.weight.shape[0]

    new_tokens = [name_tok(s) for s in ds_tr.sensors] + [name_tok(c) for c in ds_tr.class_names]

    # Añade y obtiene cuántos se han añadido de verdad
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    #tokenizer.add_tokens(new_tokens, special_tokens=True)

    # Solo resize si realmente has añadido algo (evita encoger!)
    if n_added > 0:
        llama.resize_token_embeddings(old_vocab + n_added, mean_resizing=False)
        
    print("Vocab size:", llama.get_input_embeddings().weight.shape[0])
    print("Tokenizer size:", len(tokenizer))
    
    with torch.no_grad():
        in_w = llama.get_input_embeddings().weight
        out_w = llama.get_output_embeddings().weight

        # ids de TODOS tus tokens especiales (clases + sensores)
        special_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(t) for t in new_tokens],
            device=in_w.device, dtype=torch.long
        )

        # init tipo normal pequeño (suficiente para romper el sim=1)
        std = 0.02
        in_w[special_ids].normal_(mean=0.0, std=std)

        if out_w is not in_w:
            out_w[special_ids].normal_(mean=0.0, std=std)


    # ids reales (algunos pueden existir ya si estaban en vocab)
    tok_ids = [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]


    print("tokenizer.convert_tokens_to_ids(t)")
    for t in [name_tok(c) for c in ds_tr.class_names[:5]]:
        print(t, tokenizer.convert_tokens_to_ids(t), tokenizer.encode(t, add_special_tokens=False))


    # --- LoRA AFTER resize ---
    if LORA:
        # --- LoRA config: SOLO últimas capas ---
        num_layers = int(llama.config.num_hidden_layers)
        last_n = 4
        layers_to_transform = list(range(num_layers - last_n, num_layers))

        lora_cfg = LoraConfig(
            r=4,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["gate_proj", "up_proj", "down_proj"],
            layers_to_transform=layers_to_transform,
        )

        from peft import prepare_model_for_kbit_training
        llama = prepare_model_for_kbit_training(llama)
        llama = get_peft_model(llama, lora_cfg)
        llama.print_trainable_parameters()

    # 1) Congela todo backbone
    llama.requires_grad_(False)

    # 2) Selecciona ids de tokens especiales (clases + sensores)
    special_tokens = [name_tok(s) for s in ds_tr.sensors] + [name_tok(c) for c in ds_tr.class_names]
    special_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(t) for t in special_tokens],
        device=llama.device, dtype=torch.long
    )

    def mask_grad_rows(grad):
        mask = torch.zeros_like(grad)
        mask.index_fill_(0, special_ids, 1)
        return grad * mask

    # 3) Activa grad SOLO en las matrices de embedding, pero filtrado por hook
    in_w  = llama.get_input_embeddings().weight
    out_w = llama.get_output_embeddings().weight

    in_w.requires_grad_(True)
    in_w.register_hook(mask_grad_rows)

    if out_w is not in_w:
        out_w.requires_grad_(True)
        out_w.register_hook(mask_grad_rows)





    # injector + head
    injector = UnifiedIMUInjector(llama, tokenizer, ds_tr.sensors, ds_tr.class_names, ds_tr.d_ch, ds_tr.d_sig).to(device)
    
    with torch.no_grad():
        emb_w = llama.get_input_embeddings().weight
        cls = emb_w[injector.id_sig.to(emb_w.device)].float()
        cls = F.normalize(cls, dim=-1)
        sim = cls @ cls.T
        print("sim", sim)
    print("id_sig min/max:", injector.id_sig.min().item(), injector.id_sig.max().item())
    print("id_sig unique:", len(torch.unique(injector.id_sig)))
    bad = (injector.id_sig < 0).sum().item()
    print("id_sig <0:", bad)

    head = HARHead(llama.config.hidden_size, len(ds_tr.class_names)).to(device)
    
    
    for param in injector.parameters():
        if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            param.requires_grad_(True)
    
    # Opcionales para mejorar la estructura de respuesta
    #llama.get_input_embeddings().requires_grad_(True) 
    #llama.lm_head.requires_grad_(True)

    if not LORA:
        # Cogemos todos los que requieren gradiente (llama + injector + head) de una sola vez
        param_groups = [
            {'params': list(injector.ch_proj.parameters()) + list(injector.sig_proj.parameters()), 'lr': 1e-4},
            {'params': head.parameters(), 'lr': 1e-4},
            {'params': [in_w], 'lr': 1e-5},
        ]
        if out_w is not in_w:
            param_groups.append({'params': [out_w], 'lr': 1e-5})

        opt = torch.optim.AdamW(param_groups)
    else:
        in_w  = llama.get_input_embeddings().weight
        out_w = llama.get_output_embeddings().weight

        lora_params = [p for n, p in llama.named_parameters()
                       if p.requires_grad and "lora_" in n]

        param_groups = [
            {'params': list(injector.ch_proj.parameters()) + list(injector.sig_proj.parameters()), 'lr': 1e-4},
            {'params': head.parameters(), 'lr': 1e-4},
            {'params': lora_params, 'lr': 2e-4},
            {'params': [in_w], 'lr': 1e-5},
        ]

        # ✅ solo si NO están atados
        if out_w is not in_w:
            param_groups.append({'params': [out_w], 'lr': 1e-5})

        opt = torch.optim.AdamW(param_groups)

    crit_har, crit_lm = nn.CrossEntropyLoss(), nn.CrossEntropyLoss(ignore_index=-100)

    tok = "<Frontal elevation of arms>"
    print("token->id:", tokenizer.convert_tokens_to_ids(tok))
    print("encode:", tokenizer.encode(tok, add_special_tokens=False))
    
    tok = name_tok("Frontal elevation of arms")
    print("token->id:", tokenizer.convert_tokens_to_ids(tok))
    print("encode:", tokenizer.encode(tok, add_special_tokens=False))
    
    # --- DENTRO DEL BUCLE DE ENTRENAMIENTO (train) ---

    for epoch in range(1, 40):
        llama.train(); injector.train(); head.train()
        pbar = tqdm(DataLoader(ds_tr, batch_size=4, shuffle=True))
        first_batch = True # Bandera para debug
        
        



        for batch in pbar:
            opt.zero_grad()
            X_ch, X_sig, y = batch["X_ch"].to(device), batch["X_sig"].to(device), batch["y"].to(device)
            
            prefix_len = injector.get_prefix_len()
            embeds, mask, input_ids = injector.build_sequence(
                X_ch, X_sig, batch["text"]
            )
            
            har_mask = torch.zeros_like(mask)
            har_mask[:, :prefix_len] = 1 
                        
            # --- 🟢 DEBUG BATCH 1 (Corregido p_len -> prefix_len) ---
            if first_batch:
                print(f"\n[DEBUG BATCH 1]")
                print(f"  - input_ids shape: {input_ids.shape}") 
                print(f"  - prefix_len: {prefix_len}")  # 🟢 Corregido
                print(f"  - embeds shape: {embeds.shape}")       
                
                print(f"\n🎲 [PRE-CHECK: MODOS ALEATORIOS PARA EPOCH {epoch}]")
                random_idx = random.randint(0, len(ds_te)-1)
                for i in range(4): 
                    # Cada acceso a ds_te[random_idx] dispara un modo (A, B, C o D) al azar
                    sample_variant = ds_te[random_idx]
                    print(f"  > Muestra {random_idx} (Variante {i+1}):")
                    print(f"    {sample_variant['text']}\n")
                print("-" * 50)
                
                
            

            # --- HAR Loss ---
            #start_ch = 1 + (len(ds_tr.class_names) * 2)
            #end_ch = start_ch + (len(ds_tr.sensors) * 2)
            #h_har = out.hidden_states[-1][:, start_ch:end_ch, :].to(torch.float32).mean(dim=1)
            # 1. Generar secuencias con el inyector
            
            
            # 2. Forward pass del LLM (obteniendo hidden states para HAR)
            out = llama(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)

            # 1) pooled_features primero
            pooled_features = sensorllm_pool(out.hidden_states, har_mask)

            if not ONLY_CHANNEL:
                K = len(ds_tr.class_names)
                pos_class = 1 + torch.arange(1, 2*K, 2, device=device)   # -> 2,4,6...
                cls_hidden = out.hidden_states[-1][:, pos_class, :]      # (B,K,H)

                if first_batch:
                    with torch.no_grad():
                        c = F.normalize(cls_hidden[0].float(), dim=-1)
                        sim_h = c @ c.T
                        print("sim_cls_hidden min/max:", sim_h.min().item(), sim_h.max().item())
            

            if first_batch:
                z = pooled_features.detach().float()
                print("[DEBUG] pooled mean/std:", z.mean().item(), z.std().item())
                print("[DEBUG] pooled per-dim std mean:", z.std(dim=0).mean().item())
                print("[DEBUG] pooled per-sample norms:", z.norm(dim=-1))

            # 4. Clasificación HAR (usando 'y' del batch)
            logits_har = head(pooled_features) 
            loss_har = crit_har(logits_har, y)
            # --- CLIP Losses (prioridad clase > canales) ---
            if not ONLY_CHANNEL:
                loss_clip_cls = clip_class_loss_cls_hidden(pooled_features, y, cls_hidden)
                C = len(ds_tr.sensors)

                # offsets dentro del sequence:
                # [BOS] + (sig_b si aplica) + ch_b + texto...
                off = 1
                if not ONLY_CHANNEL:
                    off += (len(ds_tr.class_names) * 2)  # sig_b ocupa 2K

                # en ch_b: [h_ch0, tok0, h_ch1, tok1, ...]
                pos_ch_tok = off + torch.arange(1, 2*C, 2, device=device)   # posiciones de los TOKENS
                h_ch_proj  = injector.ch_proj(X_ch).to(torch.float32)       # (B, C, H)
                h_tok      = out.hidden_states[-1][:, pos_ch_tok, :].to(torch.float32)  # (B, C, H)

                loss_clip_ch = (1.0 - F.cosine_similarity(
                    F.normalize(h_tok, dim=-1),
                    F.normalize(h_ch_proj, dim=-1),
                    dim=-1
                )).mean()
            else:
                loss_clip_cls = torch.zeros((), device=device)
                loss_clip_ch  = torch.zeros((), device=device)
            
            # --- LM Loss: Alineación y Recorte ---
            # --- LM Loss (alineado con full_emb) ---
           
            # --- LM Loss (alineado con full_emb) ---
            # --- LM Loss (alineado con full_emb) ---
            B = input_ids.size(0)
            T_text = input_ids.size(1)           # 256
            full_len = embeds.size(1)            # 1 + prefix + (T_text-1)

            logits = out.logits.to(torch.float32)              # (B, full_len, V)
            shift_logits = logits[:, :-1, :].contiguous()      # (B, full_len-1, V)

            labels_full = torch.full((B, full_len), -100, device=device, dtype=torch.long)

            # texto (sin BOS) empieza en prefix_len y tiene longitud (T_text-1)
            labels_full[:, prefix_len:prefix_len + (T_text - 1)] = input_ids[:, 1:]

            # ignora pad (pad==eos)
            labels_full[labels_full == tokenizer.pad_token_id] = -100

            # Answer-only mask: enmascara todo hasta justo después de "A:"
            a_ids = tokenizer.encode("A:", add_special_tokens=False)

            def find_subseq(haystack, needle):
                for j in range(0, len(haystack) - len(needle) + 1):
                    if haystack[j:j+len(needle)] == needle:
                        return j
                return -1

            for i in range(B):
                ids = input_ids[i].tolist()   # len=T_text
                j = find_subseq(ids, a_ids)
                if j != -1:
                    cut = j + len(a_ids)      # índice dentro de input_ids que deja el prompt incluido "A:"
                    end = prefix_len + max(cut - 1, 0)  # -1 porque labels_full usa input_ids[:,1:]
                    labels_full[i, prefix_len:end] = -100

            shift_labels = labels_full[:, 1:].contiguous()     # (B, full_len-1)

            valid = (shift_labels != -100)
            n_valid = int(valid.sum().item())

            if n_valid > 0:
                loss_lm_sum = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum"
                )
                loss_lm = loss_lm_sum / n_valid

                vocab_size = shift_logits.size(-1)
                loss_lm = loss_lm / math.log(vocab_size)
            else:
                loss_lm = torch.zeros((), device=device, dtype=torch.float32)

            if first_batch:
                print("full_len embeds:", embeds.size(1))
                print("full_len mask:", mask.size(1))
                print("full_len labels_full:", labels_full.size(1))
                print("shift_labels len:", shift_labels.size(1))
                print("n_valid LM tokens:", n_valid)





            # --- Optimización ---
            # Pesos (ajústalos según lo que te falle)
            w_har = 1.0
            w_clip_cls = 1.0   # ALTA prioridad (clase)
            w_clip_ch  = 1.0   # MEDIA prioridad (canales)
            w_lm = 1.0         # BAJA prioridad (next-token)

            total_loss = (
                w_har * loss_har +
                w_clip_cls * loss_clip_cls +
                w_clip_ch * loss_clip_ch +
                w_lm * loss_lm
            )
            
            if not torch.isnan(total_loss):
                total_loss.backward()
                if not LORA:
                    torch.nn.utils.clip_grad_norm_(injector.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(list(injector.ch_proj.parameters()) + list(injector.sig_proj.parameters()), 1.0)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(llama.parameters(), 1.0)

                opt.step()
            else:
                print("⚠️ Salto de batch por NaN detectado")
                opt.zero_grad()
            
            if ONLY_CHANNEL:
                pbar.set_description(
                    f"Ep {epoch} | HAR:{loss_har.item():.3f} | LM:{loss_lm.item():.3f}"
                )
            else:
                pbar.set_description(
                    f"Ep {epoch} | HAR:{loss_har.item():.3f} | CLIPc:{loss_clip_cls.item():.3f} | CLIPch:{loss_clip_ch.item():.3f} | LM:{loss_lm.item():.3f}"
                )

            first_batch=False

# --- EVALUACIÓN ---
        llama.eval(); injector.eval(); head.eval(); all_preds, all_labels = [], []
        print(f"\n--- Evaluando Epoch {epoch} ---")
       
        with torch.no_grad():
            for v_b in DataLoader(ds_te, batch_size=4):
                X_ch_v = v_b["X_ch"].to(device)
                X_sig_v = v_b["X_sig"].to(device)
                y_v = v_b["y"].to(device)
                v_text = v_b["text"] 
                
                # 1. Construir secuencia y obtener hidden states
                emb_v, m_v, _ = injector.build_sequence(X_ch_v, X_sig_v, v_b["text"])
                out_v = llama(inputs_embeds=emb_v, attention_mask=m_v, output_hidden_states=True)
                prefix_len_v = injector.get_prefix_len()
                har_mask_v = torch.zeros_like(m_v)
                har_mask_v[:, :prefix_len_v] = 1

                # 2. Aplicar Pooling tipo SensorLLM (idéntico al entrenamiento)
                # Usamos out_v.hidden_states que contiene todas las capas
                p_v = sensorllm_pool(out_v.hidden_states, har_mask_v)
                
                # 3. Predicción HAR
                logits_v = head(p_v)
                all_preds.extend(logits_v.argmax(dim=-1).cpu().numpy())
                all_labels.extend(y_v.cpu().numpy())

        # Métricas
        acc = accuracy_score(all_labels, all_preds)
        f1_m = f1_score(all_labels, all_preds, average='macro')
        f1_w = f1_score(all_labels, all_preds, average='weighted')

        print(f"📊 [METRICS] Acc: {acc:.4f} | F1-Macro: {f1_m:.4f} | F1-Weighted: {f1_w:.4f}")

        # --- PRUEBA CUALITATIVA CORREGIDA ---
        # --- PRUEBA CUALITATIVA 100% CONSISTENTE ---
        idx_eval = random.randint(0, len(ds_te)-1)
        sample = ds_te[idx_eval]

        full_text = sample["text"]

        # Extraemos pregunta y respuesta de forma segura
        split_marker = "A:"
        if split_marker in full_text:
            q_part, a_part = full_text.split(split_marker, 1)
            q_text = q_part + "A:"
            gt_answer = a_part.strip()
        else:
            # fallback de seguridad
            q_text = full_text
            gt_answer = ""

        
        # --- ANCLA: añade 1 token dummy para que generate devuelva sequences "cortables" ---
        emb_q, mask_q, _ = injector.build_sequence_nopad(
            sample["X_ch"].unsqueeze(0).to(device),
            sample["X_sig"].unsqueeze(0).to(device),
            [q_text]
        )

        bos = tokenizer.bos_token_id
        if bos is None:
            bos = tokenizer.eos_token_id   # en Qwen suele ser <|endoftext|>

        # input_ids dummy con la MISMA longitud que emb_q
        dummy_ids = torch.full(
            (emb_q.size(0), emb_q.size(1)),
            bos,
            device=emb_q.device,
            dtype=torch.long
        )

        gen_ids = llama.generate(
            input_ids=dummy_ids,          # 👈 fuerza “prompt length” en sequences
            inputs_embeds=emb_q,          # 👈 pero realmente condicionas con embeds
            attention_mask=mask_q,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            pad_token_id=tokenizer.pad_token_id
        )

        input_len = emb_q.shape[1]
        new_ids = gen_ids[0][input_len:]  # 👈 AHORA sí: prompt_len consistente
        model_a = tokenizer.decode(new_ids, skip_special_tokens=False)
        model_a = model_a.split("<|endoftext|>")[0].strip()

        print("gen_ids first 10:", gen_ids[0][:10].tolist())
        print("decoded first 30:", tokenizer.decode(gen_ids[0][:30], skip_special_tokens=False))
        print("cut:", cut, "gen_len:", gen_ids.shape[1])
        print("first_new_id:", int(gen_ids[0][cut].item()) if gen_ids.shape[1] > cut else None)
        print("eos_id:", tokenizer.eos_token_id, "pad_id:", tokenizer.pad_token_id)
        
        
        print("cut:", cut, "gen_len:", gen_ids.shape[1])
        print("pad_id:", tokenizer.pad_token_id, "eos_id:", tokenizer.eos_token_id)

        print(f"\n🔎 [CHECK CUALITATIVO - EPOCH {epoch}]")
        print(f"  Act: <{ds_te.id2label[sample['y']]}>")
        print(f"  Q:   {q_text.replace('Q: ', '').replace('\\n', ' ').strip()}")
        print(f"  GT:  {gt_answer}")
        print(f"  LLM: {model_a}")
        print("-" * 50)

        # --- GUARDADO PERIÓDICO (Cada 5 Epochs) ---
        if epoch % 5 == 0:
            save_checkpoint_custom(
                out_dir="outputs_multihar",  # Carpeta principal
                epoch=epoch,
                tokenizer=tokenizer,
                llama=llama,
                injector=injector,
                head=head,
                extra={
                    "feat_names": ds_tr.feat_names,
                    "id2label": ds_tr.id2label,
                    "lambda_lm": 5.0, # El peso que estás usando
                    "prefix_len": injector.get_prefix_len()
                },
            )

if __name__ == "__main__": train()