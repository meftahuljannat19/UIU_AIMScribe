# UIU_AIMScribe: Fine-Tuning a Large Language Model for Medical Chat Summarization

## 📘 Project Overview

This repository demonstrates **instruction-style supervised fine-tuning (SFT)** for a Llama model using lightweight adapters.  
It’s designed for reproducibility, efficiency, and clarity — perfect for domain adaptation (e.g., medical, legal, or bilingual datasets).

---

### 🔍 Key Features

- 🧠 **LoRA/QLoRA** adapter fine-tuning — minimal VRAM and training time  
- 💬 Support for **chat-style** datasets  
- 📊 Automatic evaluation with **ROUGE**, **BLEU** and **BertScore**    

---
### ⚙️ Installation Guide
Use the following code to install the dependencies 
```
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    import torch; v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
    xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
    !pip install --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
```
Use the following code to install the evaluation metrics
```
!pip install evaluate sacrebleu rouge_score bert_score
```

## 🧠 How to Think & Approach the Problem

A clear mental framework helps structure your fine-tuning workflow — from defining goals to safe deployment.  
Follow this **step-by-step strategy** 👇

---

### 🪜 Step-by-Step Framework

#### 1️⃣ Define the Goal
- 🎯 **Clarify the task:** summarization, dialogue generation, translation, or domain adaptation.  
- 📏 **Choose evaluation metrics:** ROUGE, BLEU, BertScore
- 🧩 **Set success criteria:** accuracy, fluency, and alignment with domain tone.

---

#### 2️⃣ Prepare the Data
- 🧹 **Prepare chat:** Define a chat template and formatting function for fine-tuning   
- 🔀 **Split the data fairly:** create train / test splits 

---

#### 3️⃣ Select the Strategy
- ⚙️ **Start simple:** begin with **LoRA** for efficient adapter-based training.  
- 💾 **Optimize for hardware:** use **QLoRA** if GPU memory is limited (4-bit quantization).  
- 🧮 **Tune hyperparameters:** rank `r = 8–16`, dropout `0.05`, and moderate learning rate (e.g., `2e-4`).

---

#### 4️⃣ Train Systematically
- 🧰 Use **TRL’s `SFTTrainer`** for supervised fine-tuning.  
- 🚀 Enable **gradient checkpointing** and **mixed precision (bf16/fp16)** to save memory.  
- 💾 **Save checkpoints periodically** and monitor loss and validation metrics.  
- 📈 **Visualize progress** via TensorBoard or Weights & Biases (`wandb`).

---

#### 5️⃣ Evaluate Thoughtfully
- 🔢 **Quantitative metrics:** compute ROUGE, BLEU, and other automated scores.  
- 👁️ **Qualitative review:** check fluency, factual accuracy, tone, and domain appropriateness.  
- 🧠 **Error analysis:** categorize common mistakes to guide data or prompt improvements.

---


## 📊 Evaluation Results

After fine-tuning the model, the following evaluation metrics were obtained on the validation dataset:

| Metric | Score |
|:--------|:------:|
| 🟥 **ROUGE-1** | **0.5894** |
| 🟧 **ROUGE-2** | **0.2961** |
| 🟨 **ROUGE-L** | **0.4106** |
| 🟩 **BLEU** | **0.2108** |
| 🟦 **BERTScore (F1)** | **0.8916** |

---

### 🧾 Interpretation

- **ROUGE-1 / ROUGE-2 / ROUGE-L:** indicate strong overlap between generated and reference summaries, showing good content retention.  
- **BLEU:** measures n-gram precision — moderate score suggests linguistic diversity with room for refinement.  
- **BERTScore:** very high semantic similarity (**0.89+**), confirming strong contextual alignment with reference text.

---

### 📈 Summary

✅ The fine-tuned model performs **consistently well across all metrics**, achieving a strong balance between lexical accuracy and semantic faithfulness.  
💡 Future iterations could focus on:
- Increasing dataset diversity to further improve BLEU precision.  
- Fine-tuning with a slightly longer context window for better discourse coherence.  
- Human evaluation to complement automatic metrics.

---

> 🧠 **Tip:** For transparent reporting, always specify dataset split, number of examples evaluated, and metric computation details (e.g., stemming, tokenization policy).

