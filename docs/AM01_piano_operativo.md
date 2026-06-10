# Piano operativo — 2026/AM01 Detection of Anomalous Behaviour in Industrial Robot

## 0. Scopo del progetto

Il progetto consiste nello sviluppo e nella valutazione critica di un **Adversarial Autoencoder (AAE)** per la rilevazione di anomalie in serie temporali provenienti da un robot industriale Kuka. I segnali disponibili includono sensori cinematici ed elettrici, come posizioni angolari dei giunti, velocità, correnti e consumo di potenza.

L’obiettivo non è soltanto ottenere un buon modello di anomaly detection, ma rispondere in modo sperimentale alla domanda centrale:

> L’aggiunta della componente avversaria migliora davvero la capacità di individuare comportamenti anomali rispetto a un autoencoder tradizionale?

La risposta dovrà essere supportata da una pipeline riproducibile, da baseline corrette, da metriche adatte alle time-series e da un’analisi critica dei risultati.

---

## 1. Domande di ricerca

Il progetto deve essere organizzato attorno a poche domande sperimentali chiare.

### RQ1 — Anomaly detection reconstruction-based
Un autoencoder addestrato solo su movimenti normali del robot riesce a riconoscere deviazioni nei run anomali tramite errore di ricostruzione?

### RQ2 — Valore della regolarizzazione avversaria
Un AAE, che forza il latent space ad approssimare una distribuzione prior, produce rappresentazioni più regolari e anomaly score più discriminativi rispetto a un AE standard?

### RQ3 — Robustezza rispetto alle scelte di modellazione
Le performance dipendono in modo sensibile da lunghezza della finestra temporale, stride, latent dimension, architettura, loss di ricostruzione e soglia decisionale?

### RQ4 — Valutazione rigorosa
Le conclusioni rimangono valide usando metriche point-wise, event-aware e protocolli che evitano stime ottimistiche delle performance?

---

## 2. Assunzioni iniziali da verificare appena si riceve il dataset

Prima di implementare il modello definitivo, bisogna verificare:

1. Formato dei file: CSV, NumPy, parquet, HDF5 o altro.
2. Presenza di timestamp o campionamento regolare implicito.
3. Numero di run, traiettorie o sequenze indipendenti.
4. Numero e nome dei sensori.
5. Frequenza di campionamento.
6. Presenza di label point-wise, segment-wise o solo run-level.
7. Quali run sono normali e quali sono anomali.
8. Se le anomalie sono rallentamenti, imprecisioni, offset, rumore, drift o combinazioni.
9. Eventuali NaN, outlier fisicamente impossibili, duplicati, discontinuità temporali.
10. Se train/test split è già fornito dal project owner o va definito dal gruppo.

Queste informazioni vanno documentate nel report, perché condizionano completamente il disegno sperimentale.

---

## 3. Struttura consigliata della repository

```text
am01-kuka-aae-anomaly-detection/
│
├── README.md
├── pyproject.toml / requirements.txt
├── .gitignore
├── configs/
│   ├── ae_mlp.yaml
│   ├── aae_mlp.yaml
│   ├── ae_conv1d.yaml
│   └── evaluation.yaml
│
├── data/
│   ├── raw/                 # non versionare se troppo pesante
│   ├── processed/           # finestre, scaler, split
│   └── README_data.md       # descrizione dataset e istruzioni
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda_signals.ipynb
│   ├── 03_baseline_ae.ipynb
│   ├── 04_aae_debug.ipynb
│   └── 05_results_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── loading.py
│   │   ├── preprocessing.py
│   │   ├── windowing.py
│   │   └── splits.py
│   │
│   ├── models/
│   │   ├── ae.py
│   │   ├── aae.py
│   │   ├── discriminator.py
│   │   └── baselines.py
│   │
│   ├── training/
│   │   ├── train_ae.py
│   │   ├── train_aae.py
│   │   ├── losses.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/
│   │   ├── scoring.py
│   │   ├── thresholding.py
│   │   ├── metrics.py
│   │   └── plots.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── io.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiments.py
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── runs/
│
├── report/
│   ├── main.tex
│   ├── sections/
│   └── figures/
│
└── presentation/
    └── slides.pptx
```

La repository deve mostrare autonomia, ordine e riproducibilità. I notebook servono per esplorare; il codice definitivo deve stare in `src/` e `scripts/`.

---

## 4. Pipeline generale

```text
Raw robot time-series
        ↓
Data audit + EDA
        ↓
Cleaning minimale e normalizzazione
        ↓
Sliding windows temporali
        ↓
Train/validation/test split senza leakage
        ↓
Baseline non deep learning
        ↓
Autoencoder tradizionale
        ↓
Adversarial Autoencoder
        ↓
Anomaly scoring
        ↓
Threshold selection su validation
        ↓
Valutazione finale su test
        ↓
Analisi critica AE vs AAE
```

---

## 5. Work Package 1 — Data audit ed esplorazione

### Obiettivo
Capire esattamente cosa contiene il dataset e quali difficoltà reali presenta.

### Attività

1. Caricare tutti i file grezzi.
2. Costruire una tabella riepilogativa per ogni run:
   - ID run;
   - lunghezza temporale;
   - numero di sensori;
   - label disponibile;
   - percentuale di NaN;
   - media/deviazione standard per sensore;
   - eventuali anomalie note.
3. Visualizzare i segnali principali:
   - joint positions;
   - velocities;
   - current;
   - power usage.
4. Verificare differenze macroscopiche tra run normali e anomali.
5. Studiare correlazioni tra sensori.
6. Verificare se alcune variabili sono ridondanti, costanti o quasi costanti.
7. Salvare figure utili per report e presentazione.

### Output atteso

- Notebook `01_data_audit.ipynb`.
- Tabella `dataset_summary.csv`.
- Figure iniziali dei segnali.
- Breve documento `data/README_data.md`.

### Criterio di completamento
Il gruppo deve essere in grado di spiegare a parole cosa rappresenta ogni dimensione del dato e quali run sono usati per training, validation e test.

---

## 6. Work Package 2 — Preprocessing e windowing

### Obiettivo
Trasformare le serie temporali grezze in campioni adatti all’addestramento di AE e AAE.

### Scelte operative

#### 6.1 Normalizzazione
Usare uno scaler calcolato solo sul training set normale. Opzioni:

- `StandardScaler`: scelta iniziale consigliata.
- `RobustScaler`: utile se i segnali normali contengono outlier.
- Min-max scaling: meno robusto, da usare solo se motivato.

Regola fondamentale: nessuna informazione del test deve entrare nello scaler.

#### 6.2 Sliding windows
Convertire la serie temporale multivariata in finestre:

```text
X_window ∈ R^{T × C}
```

dove:

- `T` = lunghezza temporale della finestra;
- `C` = numero di canali/sensori.

Configurazioni iniziali da provare:

| Configurazione | Window length | Stride | Uso |
|---|---:|---:|---|
| breve | 32 | 8 | anomalie rapide |
| media | 64 | 16 | default iniziale |
| lunga | 128 | 32 | rallentamenti e drift |

La scelta finale va fatta tramite validation, non osservando il test.

#### 6.3 Label di finestra
Se sono disponibili label point-wise, una finestra può essere etichettata come anomala se contiene almeno un punto anomalo. In alternativa si può usare una soglia, per esempio finestra anomala se almeno il 10% dei punti è anomalo.

Da riportare nel report:

- criterio usato;
- motivazione;
- impatto sul numero di finestre normali/anomale.

### Output atteso

- `processed_train.pt` / `.npz`
- `processed_val.pt` / `.npz`
- `processed_test.pt` / `.npz`
- scaler serializzato
- configurazione di preprocessing in YAML

---

## 7. Work Package 3 — Protocollo di valutazione

### Obiettivo
Definire la valutazione prima di ottimizzare i modelli, evitando conclusioni gonfiate o leakage.

### Regole principali

1. Addestrare AE e AAE solo su dati normali.
2. Usare validation per scegliere soglie e iperparametri.
3. Usare test solo una volta per il confronto finale.
4. Non scegliere la soglia sul test.
5. Evitare protocolli eccessivamente permissivi, come point adjustment non controllato.
6. Riportare sia metriche point-wise/window-wise sia metriche event-aware.

### Metriche consigliate

#### Ranking metrics, senza scelta di soglia

- ROC-AUC.
- PR-AUC, particolarmente importante se le anomalie sono rare.

#### Metriche con soglia

- Precision.
- Recall.
- F1-score.
- Balanced accuracy.
- False positive rate.
- False alarms per run o per unità temporale.

#### Metriche temporali/event-aware

- Event recall: percentuale di eventi anomali individuati almeno una volta.
- Detection delay: ritardo tra inizio evento e prima rilevazione.
- Event precision: quanti eventi predetti corrispondono a eventi reali.
- Durata dei falsi allarmi.

### Output atteso

- File `evaluation_protocol.md`.
- Funzioni riutilizzabili in `src/evaluation/metrics.py`.
- Script `scripts/evaluate.py`.

---

## 8. Work Package 4 — Baseline obbligatorie

Prima dell’AAE bisogna costruire baseline solide. Senza baseline, non è possibile sostenere che la componente avversaria migliori davvero il risultato.

### Baseline 1 — Reconstruction PCA
Una PCA addestrata sui dati normali, con ricostruzione e anomaly score basato su errore di ricostruzione.

Motivo: baseline semplice, interpretabile, molto utile per capire se il problema è già quasi lineare.

### Baseline 2 — Isolation Forest o Local Outlier Factor
Da applicare su feature di finestra, per esempio media, deviazione standard, minimo, massimo, energia, differenze temporali.

Motivo: nei laboratori viene suggerito che, per dati tabulari o feature-based, metodi non deep possono essere competitivi e rapidi.

### Baseline 3 — Autoencoder MLP
Input flatten della finestra:

```text
T × C → T*C
```

Architettura iniziale:

```text
input_dim → 256 → 128 → latent_dim → 128 → 256 → input_dim
```

### Baseline 4 — Autoencoder temporale
Almeno una variante che rispetti meglio la natura sequenziale:

- Conv1D Autoencoder, consigliato come prima scelta;
- LSTM Autoencoder, più pesante ma coerente con la parte RNN/LSTM del corso.

### Baseline 5 — Untrained / weak baseline
Valutare un modello non addestrato o una scoring rule banale può essere utile come sanity check, specialmente per evitare di sovrainterpretare piccoli miglioramenti.

---

## 9. Work Package 5 — Autoencoder tradizionale

### Obiettivo
Costruire il riferimento principale contro cui confrontare l’AAE.

### Formulazione

Dato un input `x`, l’autoencoder produce:

```text
z = Encoder(x)
x_hat = Decoder(z)
```

Loss:

```text
L_AE = L_rec(x, x_hat)
```

Loss candidate:

- MSE: default iniziale.
- MAE: più robusta a outlier.
- Huber loss: compromesso tra MSE e MAE.

### Anomaly score

Default:

```text
score(x) = mean((x - x_hat)^2)
```

Varianti utili:

- errore medio per finestra;
- errore massimo per finestra;
- errore pesato per gruppi di sensori;
- errore separato per posizioni, velocità, corrente, potenza.

### Cosa salvare

- loss train/validation;
- distribuzione degli errori su train, validation normale, validation anomala, test;
- curve ROC e PR;
- soglia selezionata;
- esempi visuali di finestre ricostruite bene/male.

---

## 10. Work Package 6 — Adversarial Autoencoder

### Obiettivo
Implementare un AAE che mantenga la capacità ricostruttiva dell’autoencoder ma regolarizzi il latent space tramite training avversario.

### Componenti

1. Encoder `E(x)`.
2. Decoder `D(z)`.
3. Discriminator `Q(z)` che distingue:
   - campioni reali dal prior `p(z)`, per esempio `N(0, I)`;
   - codifiche prodotte dall’encoder `E(x)`.

### Training in tre fasi

#### Fase A — Reconstruction phase
Aggiornare encoder e decoder per minimizzare la ricostruzione:

```text
z = E(x)
x_hat = D(z)
L_rec = reconstruction_loss(x, x_hat)
```

#### Fase B — Discriminator phase
Aggiornare il discriminator per distinguere latent reali e latent codificati:

```text
z_real ~ p(z)
z_fake = E(x).detach()
L_disc = BCE(Q(z_real), 1) + BCE(Q(z_fake), 0)
```

#### Fase C — Adversarial encoder phase
Aggiornare l’encoder per ingannare il discriminator:

```text
z_fake = E(x)
L_adv = BCE(Q(z_fake), 1)
```

Loss totale dell’encoder:

```text
L_encoder = L_rec + λ_adv * L_adv
```

### Anomaly score dell’AAE

Score principale:

```text
score_rec(x) = reconstruction_error(x, D(E(x)))
```

Score opzionale:

```text
score_latent(x) = -log Q(E(x))
```

Score combinato, da usare solo se validato:

```text
score_total = α * normalize(score_rec) + (1 - α) * normalize(score_latent)
```

La versione principale del confronto deve rimanere pulita: AE vs AAE con lo stesso anomaly score ricostruttivo. Lo score combinato può essere un’analisi aggiuntiva.

---

## 11. Work Package 7 — Esperimenti e ablation study

### Esperimento minimo accettabile

1. PCA reconstruction.
2. Isolation Forest o LOF.
3. AE MLP.
4. Conv1D-AE o LSTM-AE.
5. AAE con stessa struttura encoder-decoder dell’AE principale.

### Esperimenti consigliati

| Esperimento | Scopo |
|---|---|
| AE vs AAE con stessa architettura | isolare l’effetto della componente avversaria |
| window length sweep | capire la scala temporale delle anomalie |
| latent dimension sweep | valutare compressione vs ricostruzione |
| loss MSE/MAE/Huber | robustezza rispetto a outlier |
| prior normale vs mixture prior | capire se il latent space normale è unimodale o multimodale |
| λ_adv sweep | capire quanto pesa la regolarizzazione avversaria |
| score reconstruction vs combined score | capire se il discriminator aiuta anche in inference |

### Griglia iniziale realistica

```text
window_length: [32, 64, 128]
latent_dim: [4, 8, 16, 32]
loss: [MSE, Huber]
lambda_adv: [0.01, 0.1, 1.0]
architecture: [MLP, Conv1D]
seed: [0, 1, 2]
```

Non serve testare tutto subito. Prima si costruisce una pipeline funzionante con una configurazione default; poi si allarga la ricerca.

---

## 12. Work Package 8 — Analisi dei risultati

### Tabelle obbligatorie

#### Tabella 1 — Dataset summary

| Split | Run | Normale/anomalo | N. finestre | % anomalie | Note |
|---|---|---:|---:|---:|---|

#### Tabella 2 — Modelli

| Modello | Architettura | Window | Latent | Score | Parametri |
|---|---|---:|---:|---|---:|

#### Tabella 3 — Performance finale

| Modello | ROC-AUC | PR-AUC | Precision | Recall | F1 | Event Recall | Delay medio | Falsi allarmi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

#### Tabella 4 — Ablation AAE

| λ_adv | Latent dim | PR-AUC | F1 | Note |
|---:|---:|---:|---:|---|

### Figure obbligatorie

1. Esempio di segnali normali e anomali.
2. Distribuzione reconstruction error: train normale vs test normale vs test anomalo.
3. ROC curve.
4. Precision-recall curve.
5. Timeline con ground truth e anomaly score.
6. Latent space AE vs AAE, se latent dimension 2 o tramite PCA/UMAP.
7. Esempi di finestre ricostruite correttamente e male.

### Discussione critica

La discussione non deve limitarsi a dire “AAE è meglio” o “AAE è peggio”. Deve rispondere a:

- Il miglioramento è stabile su più seed?
- Il miglioramento è visibile su PR-AUC, non solo su ROC-AUC?
- L’AAE migliora la separazione degli anomaly score?
- L’AAE ricostruisce peggio o meglio i normali?
- La componente avversaria rende il training instabile?
- Il discriminator aggiunge valore in inference?
- Le anomalie lente vengono rilevate con ritardo?
- Ci sono molti falsi positivi durante transizioni normali del robot?

---

## 13. Collegamento con i laboratori del corso

La cartella dei laboratori è utile come base concettuale e implementativa.

### LabsAM

- `Lab1 / vae_pytorch.py`: utile per capire encoder, decoder, latent space, campionamento e regolarizzazione.
- `Lab2 / anomaly_pytorch.py`: base minimale per reconstruction-based anomaly detection con autoencoder addestrato su dati normali.
- `Lab3 / gan_pytorch.py`: utile per impostare generator/discriminator e training avversario.
- `Lab4 / diffusion_pytorch.py`: meno centrale per AM01, ma utile per il ragionamento su modelli generativi e distribuzioni.

### LabsFP

- Lab time-series anomaly detection: utile per impostare pipeline sklearn, scaling, PCA, SVM, metriche, confusion matrix e hyperparameter tuning.
- Lab confidence/uncertainty: utile come possibile estensione per interpretare casi incerti o out-of-distribution.

Nel progetto finale non bisogna copiare i laboratori così come sono. Bisogna trasformare quelle idee in una pipeline modulare, riproducibile e adatta al dataset Kuka.

---

## 14. Report scientifico

Struttura consigliata:

1. **Abstract**
   - problema;
   - metodo;
   - confronto AE/AAE;
   - risultato principale.

2. **Introduction**
   - robot industriali e anomaly detection;
   - motivazione: configurazione errata, aging, rallentamenti, perdita di precisione;
   - contributo del lavoro.

3. **Background**
   - time-series anomaly detection;
   - reconstruction-based anomaly detection;
   - autoencoder;
   - adversarial autoencoder;
   - criticità della valutazione.

4. **Dataset**
   - descrizione segnali;
   - split;
   - preprocessing;
   - windowing;
   - statistiche.

5. **Methods**
   - baseline non deep;
   - AE;
   - AAE;
   - anomaly score;
   - thresholding.

6. **Experimental protocol**
   - train/validation/test;
   - metriche;
   - gestione delle soglie;
   - semi di randomizzazione;
   - hardware/software.

7. **Results**
   - tabelle;
   - curve;
   - timeline;
   - ablation.

8. **Discussion**
   - cosa funziona;
   - cosa non funziona;
   - quando l’AAE è utile;
   - limiti del lavoro.

9. **Conclusions**
   - risposta finale alla research question;
   - possibili sviluppi.

10. **Appendix**
   - iperparametri;
   - dettagli architetture;
   - configurazioni complete.

---

## 15. Presentazione da 20 minuti

Struttura consigliata: 12-14 slide.

| Slide | Contenuto | Tempo |
|---|---|---:|
| 1 | Titolo, team, progetto AM01 | 30s |
| 2 | Problema industriale e obiettivo | 1m |
| 3 | Dataset Kuka e sensori | 1.5m |
| 4 | EDA: segnali normali/anomali | 1.5m |
| 5 | Pipeline generale | 1m |
| 6 | Autoencoder baseline | 1.5m |
| 7 | Adversarial Autoencoder | 2m |
| 8 | Protocollo di valutazione | 2m |
| 9 | Risultati principali | 2m |
| 10 | AE vs AAE: confronto critico | 2m |
| 11 | Ablation e failure cases | 2m |
| 12 | Conclusioni | 1m |
| 13 | Backup: architetture/iparametri | backup |
| 14 | Backup: metriche e soglie | backup |

La presentazione deve far vedere che avete capito il problema, non solo che avete addestrato una rete.

---

## 16. Timeline consigliata

### Settimana 1 — Dataset e baseline minimale

- Caricamento dataset.
- EDA.
- Preprocessing base.
- Sliding windows.
- Autoencoder minimale funzionante.

Deliverable: primo notebook end-to-end.

### Settimana 2 — Protocollo e baseline solide

- Split definitivo.
- Metriche.
- PCA reconstruction.
- Isolation Forest/LOF.
- AE MLP pulito.

Deliverable: prima tabella di risultati.

### Settimana 3 — Modello temporale

- Conv1D-AE o LSTM-AE.
- Tuning window length.
- Analisi reconstruction error.

Deliverable: confronto AE MLP vs AE temporale.

### Settimana 4 — AAE

- Implementazione discriminator.
- Training a tre fasi.
- Debug stabilità.
- Primo confronto AE vs AAE.

Deliverable: AAE funzionante con curve di training.

### Settimana 5 — Ablation e valutazione rigorosa

- Sweep su latent dimension e λ_adv.
- Multi-seed.
- Metriche event-aware.
- Figure definitive.

Deliverable: risultati finali congelati.

### Settimana 6 — Report e presentazione

- Scrittura report.
- Preparazione slide.
- Pulizia repo.
- README e istruzioni di riproduzione.

Deliverable: materiale pronto almeno una settimana prima dell’esame.

---

## 17. Rischi principali e mitigazioni

| Rischio | Effetto | Mitigazione |
|---|---|---|
| Poche anomalie | metriche instabili | usare PR-AUC, event recall, analisi qualitativa |
| Leakage tra finestre sovrapposte | performance gonfiate | split per run prima del windowing |
| Soglia scelta sul test | valutazione non valida | scegliere soglia solo su validation |
| AAE instabile | risultati peggiori di AE | ridurre λ_adv, semplificare discriminator, monitorare training |
| Anomalie lente | detection delay elevato | finestre più lunghe, metriche event-aware |
| Sensori con scale diverse | loss dominata da pochi canali | standardizzazione per canale, loss pesata |
| AE ricostruisce anche anomalie | bassa separazione | bottleneck più stretto, regolarizzazione, early stopping |
| ROC-AUC alta ma falsi allarmi elevati | sistema poco utile | PR-AUC, FPR, false alarms per run |

---

## 18. Definition of Done

Il progetto può considerarsi completo quando sono disponibili:

1. Dataset preprocessing riproducibile.
2. Split documentato e privo di leakage.
3. Almeno due baseline non AAE.
4. Autoencoder tradizionale funzionante.
5. Adversarial Autoencoder funzionante.
6. Confronto AE vs AAE a parità di condizioni.
7. Metriche point-wise/window-wise e almeno una metrica event-aware.
8. Figure chiare dei segnali, degli score e delle ricostruzioni.
9. Report in stile paper.
10. Presentazione da massimo 20 minuti.
11. Repository ordinata, privata, con README e comandi per riprodurre gli esperimenti.

---

## 19. Prima milestone concreta

La prima milestone deve essere volutamente semplice:

> Entro pochi giorni dalla ricezione del dataset, ottenere una pipeline che carica i dati, crea finestre, normalizza usando solo il train normale, addestra un autoencoder minimale e produce una curva di anomaly score confrontata con le label disponibili.

Questa milestone non serve a ottenere il miglior risultato, ma a verificare che l’intero flusso funzioni. Dopo questa, il progetto diventa un problema di miglioramento controllato e confronto scientifico.

---

## 20. Aggiornamento operativo approvato — esecuzione Colab-first

Questa sezione registra le decisioni operative adottate dopo l’ispezione del dataset reale e del repository. Ogni modifica successiva alla direzione del progetto deve essere riflessa qui o nei documenti collegati, in modo che codice, notebook e piano restino allineati.

### 20.1 Dataset reale

Il dataset reale disponibile non è un CSV già pronto, ma una directory NumPy:

```text
data/raw/KukaVelocityDataset/
├── KukaColumnNames.npy
├── KukaNormal.npy
└── KukaSlow.npy
```

Decisioni operative:

1. `KukaNormal.npy` viene caricato come dati normali con `label=0`.
2. `KukaSlow.npy` contiene la colonna `anomaly`, convertita in `label` binaria.
3. `action` è trattata come metadato e non come feature sensoriale, per evitare shortcut o leakage.
4. I `run_id` vengono creati dai segmenti contigui di `action`, strategia da verificare tramite data audit.
5. Le feature inferite escludono `run_id`, `t`, `label`, `anomaly`, `action` e `source_file`.

### 20.2 Esecuzione su Colab

L’esecuzione ufficiale del progetto avverrà tramite:

```text
notebooks/AM01_colab_master.ipynb
```

Il notebook deve orchestrare la pipeline, non duplicarla. Tutta la logica riutilizzabile resta in `src/` e `scripts/`.

L’analisi visuale dei risultati già prodotti avverrà tramite:

```text
notebooks/AM01_results_analysis_colab.ipynb
```

Questo secondo notebook legge `MyDrive/AM01/results`, genera tabelle, grafici,
failure cases, curve, training history e insight automatici. Non addestra modelli.

Gli esperimenti incrementali successivi ai risultati principali avverranno tramite:

```text
notebooks/AM01_phase2_colab.ipynb
```

Questo notebook scrive solo in `MyDrive/AM01/results/phase2` e non sovrascrive i
risultati principali.

Le diagnostiche AAE-specifiche successive a Phase 1/Phase 2 avverranno tramite:

```text
notebooks/AM01_phase3_aae_latent_diagnostics_colab.ipynb
```

Questo notebook non addestra modelli: carica i checkpoint AE/AAE salvati, calcola
score latenti, score del discriminator, score combinati, analisi per `action` e
analisi per feature. Scrive solo in `MyDrive/AM01/results/phase3_aae_diagnostics`.

Layout Drive consigliato:

```text
MyDrive/AM01/
├── data/
│   ├── KukaVelocityDataset/
│   └── processed/
└── results/
    ├── data_audit/
    ├── figures/
    ├── phase2/
    ├── phase3_aae_diagnostics/
    ├── runs/
    └── tables/
```

### 20.3 Modifica della roadmap sperimentale

La roadmap scientifica originale resta valida, ma il percorso operativo viene reso più conservativo:

1. Prima data audit reale e validazione della segmentazione per `action`.
2. Poi run principale con:
   - PCA reconstruction;
   - Isolation Forest;
   - AE MLP;
   - AAE MLP;
   - Conv1D-AE.
3. Solo dopo il run principale si esegue Phase 2:
   - confronto soglie `best_f1` vs percentile sui normali di validation;
   - multi-seed stability;
   - ablation AAE su `latent_dim` e `lambda_adv`;
   - ablation preprocessing/loss su `StandardScaler`, `RobustScaler`, MSE e Huber.
4. Dopo Phase 1/Phase 2 si esegue Phase 3, dedicata alla diagnostica AAE senza retraining:
   - latent norm score;
   - Mahalanobis score nello spazio latente;
   - discriminator score;
   - combinazioni reconstruction + latent/discriminator;
   - PCA dello spazio latente;
   - analisi per `action`;
   - errore di ricostruzione per feature.

Motivazione: su Colab è preferibile ottenere prima un confronto AE/AAE pulito e riproducibile, poi allargare la ricerca. Una griglia ampia fin dall’inizio rischia di consumare tempo GPU senza produrre evidenza interpretabile.

### 20.4 Confronto principale AE vs AAE

Il confronto principale deve rimanere a parità di condizioni:

1. stessa segmentazione;
2. stesso preprocessing;
3. stesso `window_length` e `stride`;
4. stesso `latent_dim`;
5. stessa architettura encoder-decoder;
6. stesso anomaly score ricostruttivo;
7. soglia scelta solo su validation.

Lo score latent/discriminator dell’AAE è ammesso solo come analisi aggiuntiva, non come risultato principale.

### 20.5 Repository minimale

Il repository deve contenere solo:

1. codice sorgente riutilizzabile;
2. script CLI necessari alla pipeline;
3. configurazioni YAML;
4. test;
5. documentazione operativa;
6. notebook Colab master, notebook di analisi risultati, notebook Phase 2 e notebook Phase 3.

Directory vuote e `.gitkeep` non necessari vanno rimossi. Gli output generati (`results/`, `data/processed/`, checkpoint e figure) restano artefatti locali o su Drive, non parte del codice sorgente.
Script e moduli legacy non necessari alla pipeline reale vengono rimossi: i test del repository usano fixture minimali interne e il progetto operativo lavora sul dataset Kuka reale.
