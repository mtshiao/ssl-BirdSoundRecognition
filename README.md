# **Bird Sound Classifier Using Self-Supervised Learning**

This repository contains the code and resources for our bird sound classifier, which utilizes self-supervised learning for pretraining on extensive unlabeled soundscape recordings collected from 22 monitoring stations in subtropical montane forests of Taiwan. The model is subsequently fine-tuned to identify 31 bird species native to Taiwan's montane regions. 

## **Key Features**
1. **Focused on dawn chorus bird song recognition**  
   The model is designed to classify bird songs during the dawn chorus from soundscape recordings. We trained the model using recordings from this specific period and validated its effectiveness through practical inference tests.  

2. **Addressing data imbalances and cross-domain challenges**  
   By incorporating a small portion of open-source datasets and employing augmentation techniques, the model effectively handles data imbalance and domain variation issues.  

3. **Enhanced robustness with a 'NOTA' category**  
   To improve adaptability in open-set recognition tasks, a "None of the Above" (NOTA) category is introduced, enabling the model to better handle non-target sounds and background noise.  

This classifier is tailored for ecological studies and supports bird monitoring in remote subtropical montane ecosystems. For detailed information, please refer to our publication in Ecological Information.

---
## **MODEL ARCHITECTURE**
The classifier uses a transformer-based architecture with a self-supervised pretraining strategy. Below is the architecture diagram:

<div style="text-align: left;">
  <img src="./architecture_diagram.png" alt="MODEL ARCHITECTURE" width="600">
</div>

## **Pipeline**
### **Overview of the Model**
The model pipeline consists of:
1. **Pretraining** on large-scale unlabeled soundscape data using a self-supervised learning approach.
2. **Fine-tuning** on a small, labeled dataset of 31 bird species found in Taiwan's montane forests.
3. **Inference** on soundscape recordings, focusing on dawn chorus bird songs.

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

## **Citation**
Wei, Y.C., Chen, W.L., Tuanmu, M.L., Lu, S.S., Shiao, M.T., 2025. Advanced montane bird monitoring using self-supervised learning and transformer on passive acoustic data. Ecological Information.https://doi.org/10.1016/j.ecoinf.2024.102927.
