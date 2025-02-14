# **Bird Sound Classifier Using Self-Supervised Learning**

This repository provides the code and resources for our **self-supervised bird sound classifier**, developed for identifying 31 bird species in Taiwan’s subtropical montane forests. The model is pretrained on **large-scale unlabeled soundscape recordings** collected from 22 monitoring stations and fine-tuned to enhance its classification performance.

Our work is based on the **AudioMAE framework** ([GitHub](https://github.com/facebookresearch/AudioMAE)). We have specifically adapted it for bird sound classification, integrating domain-specific enhancements to address **data imbalances**, **cross-domain variability**, and **open-set recognition**.

---

## **Key Features**
### 1. **Specialized for Dawn Chorus Bird Song Recognition**
The model is designed to classify **dawn chorus** bird vocalizations in soundscape recordings. It has been trained using recordings from this critical time window and validated through real-world inference tests.

### 2. **Handling Data Imbalance & Cross-Domain Challenges**
By integrating a **small portion of open-source datasets** and applying **data augmentation techniques**, the model improves recognition across different recording conditions while mitigating **class imbalance**.

### 3. **Robust Open-Set Recognition with "NOTA"**
A **"None of the Above" (NOTA)** category is introduced to help the model distinguish **non-target sounds** (e.g., environmental noise) from actual bird vocalizations, enhancing its generalization ability.

This classifier is designed for **ecological studies** and supports **long-term bird monitoring** in remote montane ecosystems.


---

## **Model Architecture**
Our classifier is built on a **transformer-based architecture**, incorporating **self-supervised pretraining** followed by fine-tuning. Below is the model architecture diagram:

<div style="text-align: left;">
  <img src="./architecture_diagram.png" alt="MODEL ARCHITECTURE" width="600">
</div>

---

## **Pipeline**
### **Overview of the Model**
The model pipeline consists of:
1. **Pretraining** on large-scale, unlabeled soundscape data using self-supervised learning.
2. **Fine-tuning** on a labeled dataset of **31 bird species** from Taiwan’s montane forests.
3. **Inference** on soundscape recordings, focusing on dawn chorus bird songs.

---

## **FINE-TUNED BIRD SPECIES**
Below is the list of 31 bird species included in the fine-tuning process:
| **No.** | **Species ID** | **Scientific Name**         | **Chinese Name**       | **Common Name**              |
|:-------:|:--------------:|:---------------------------:|:-----------------------:|:-----------------------------:|
| 1       | AA             | *Abroscopus albogularis*    | 棕面鶯                 | Rufous-faced Warbler         |
| 2       | AC             | *Arborophila crudigularis*  | 臺灣山鷓鴣             | Taiwan Partridge             |
| 3       | AM             | *Alcippe morrisonia*        | 繡眼畫眉               | Morrison's Fulvetta          |
| 4       | BG             | *Brachypteryx goodfellowi*  | 小翼鶇                 | Taiwan Shortwing             |
| 5       | BS             | *Bambusicola sonorivox*     | 臺灣竹雞               | Taiwan Bamboo-Partridge      |
| 6       | CR             | *Cyanoderma ruficeps*       | 山紅頭                 | Rufous-capped Babbler        |
| 7       | DI             | *Dicaeum ignipectus*        | 紅胸啄花               | Fire-breasted Flowerpecker   |
| 8       | EE             | *Erythrogenys erythrocnemis*| 大彎嘴                 | Black-necklaced Scimitar-Babbler |
| 9       | FH             | *Ficedula hyperythra*       | 黃胸青鶲               | Snowy-browed Flycatcher      |
| 10      | GB             | *Taenioptynx brodiei*       | 鵂鶹                   | Collared Owlet               |
| 11      | HA             | *Heterophasia auricularis*  | 白耳畫眉               | White-eared Sibia            |
| 12      | HAC            | *Horornis acanthizoides*    | 深山鶯                 | Yellowish-bellied Bush Warbler |
| 13      | HS             | *Hierococcyx sparverioides* | 鷹鵑                   | Large Hawk-Cuckoo           |
| 14      | LS             | *Liocichla steerii*         | 黃胸藪眉               | Taiwan Liocichla             |
| 15      | MH             | *Machlolophus holsti*       | 黃山雀                 | Taiwan Yellow Tit            |
| 16      | ML             | *Myiomela leucura*          | 白尾鴝                 | White-tailed Robin           |
| 17      | NV             | *Niltava vivida*            | 黃腹琉璃               | Taiwan Vivid Niltava         |
| 18      | PA             | *Periparus ater*            | 煤山雀                 | Coal Tit                     |
| 19      | PAL            | *Pnoepyga albiventer*       | 台灣鷦眉(鱗胸鷦眉)     | Scaly-breasted Cupwing       |
| 20      | PC             | *Picus canus*               | 綠啄木                 | Gray-headed Woodpecker       |
| 21      | PM             | *Parus monticolus*          | 青背山雀               | Green-backed Tit             |
| 22      | PN             | *Psilopogon nuchalis*       | 五色鳥                 | Taiwan Barbet                |
| 23      | PNI            | *Pyrrhula nipalensis*       | 褐鷽                   | Brown Bullfinch              |
| 24      | PP             | *Pterorhinus poecilorhynchus* | 棕噪眉(竹鳥)           | Rusty Laughingthrush         |
| 25      | PS             | *Pericrocotus solaris*      | 灰喉山椒鳥             | Gray-chinned Minivet         |
| 26      | RG             | *Regulus goodfellowi*       | 火冠戴菊鳥             | Flamecrest                   |
| 27      | SB             | *Schoeniparus brunneus*     | 頭烏線                 | Dusky Fulvetta               |
| 28      | SE             | *Sitta europaea*            | 茶腹鳾                 | Eurasian Nuthatch            |
| 29      | TM             | *Trochalopteron morrisonianum* | 臺灣噪眉               | White-whiskered Laughingthrush |
| 30      | TS             | *Treron sieboldii*          | 綠鳩                   | White-bellied Green-Pigeon   |
| 31      | YB             | *Yuhina brunneiceps*        | 冠羽畫眉               | Taiwan Yuhina                |

## **MODEL CHECKPOINTS**
The following pre-trained and fine-tuned model checkpoints are available for download:
| **Checkpoint Name** | **Dataset**            | **Performance (mAP)** | **Link**            |
|----------------------|------------------------|---------------------------|--------------------|
| Pre-trained (SSL)    | Soundscapes           | N/A                        | [Download](https://drive.google.com/file/d/13e2i4smPk6wttyP41EFKv0qMb4gwZSnD/view?usp=sharing) |
| Fine-tuned           | Taiwan Montane Birds  | 85.6%                      | [Download](https://drive.google.com/file/d/1rmofMFgQfPcGUlOWdbTRXmDr5FUixc3V/view?usp=sharing) |

---

## **Setting Up the Repository**
To set up the repository and run the model, follow these steps.

### **1. Prerequisites**
- **Operating System**: Linux (Recommended)
- **Python Version**: Python 3.9
- **Conda**: Anaconda or Miniconda installed

## **2. Conda Environment Setup**
We use a **prepackaged Conda environment** based on [AudioMAE](https://github.com/facebookresearch/AudioMAE).  

### **Download the Prepackaged Environment**
- Download the Conda-packed environment provided by **AudioMAE** from [this link](https://drive.google.com/file/d/1ECVmVyscVqmhI7OQa0nghIsWVaZhZx3q/view?usp=sharing).  
- Save the file in your **Downloads** directory (`~/Downloads/`).  

### **Extract and Set Up the Environment**
Run the following commands to extract the archive and move it to your Conda environment directory automatically:

```bash
#!/bin/bash

# Extract Conda environment
mkdir -p ~/Downloads/mae && tar -xzvf ~/Downloads/mae.tar.gz -C ~/Downloads/mae

# Detect Conda installation
command -v conda &> /dev/null || { echo "Error: Conda not found. Install it first."; exit 1; }

CONDA_BASE=$(conda info --base)
CONDA_ENV_DIR="$CONDA_BASE/envs"

# Check if the 'mae' environment already exists
if conda env list | grep -q "mae"; then
    echo "Error: 'mae' environment already exists. Remove it or use a different name."
    exit 1
fi

# Move extracted environment and register it
mv ~/Downloads/mae "$CONDA_ENV_DIR/"
conda env list | grep -q "mae" || conda env update -n mae --file "$CONDA_ENV_DIR/mae/environment.yml" --prune

# Activate environment
echo "Activating 'mae'..."
source "$CONDA_BASE/bin/activate" mae
```

### **3. Running the Model**  
Once the environment is set up, you can proceed with **inference** or **fine-tuning**.

---

## **Reference**  
- **AudioMAE Repository**: [GitHub](https://github.com/facebookresearch/AudioMAE)  
- **Original Paper**:  
  > **P.Y. Huang, H. Xu, J. Li, A. Baevski, M. Auli, W. Galuba, F. Metze, C. Feichtenhofer**  
  > *Masked Autoencoders That Listen*. arXiv (2022), [10.48550/arXiv.2207.06405](https://arxiv.org/abs/2207.06405)  

---

## **Citation**  
Please cite:

> **Wei, Y.C., Chen, W.L., Tuanmu, M.L., Lu, S.S., Shiao, M.T.**  
> *Advanced montane bird monitoring using self-supervised learning and transformer on passive acoustic data.*  
> **Ecological Information (2024).**  [DOI](https://doi.org/10.1016/j.ecoinf.2024.102927)
