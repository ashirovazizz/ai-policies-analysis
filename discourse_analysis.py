# -*- coding: utf-8 -*-
r"""
Разворот проекта на ДИСКУРС-анализ.
Центр — дискурсивные кластеры AI-политик, найденные снизу вверх (BERTopic),
а не дисциплина. Дисциплина остаётся вторичной осью (кросстаб дискурс x дисциплина).

Переиспользует уже посчитанные эмбеддинги (outputs/embeddings_bge.npy) и
outputs/cleaned_policies.csv — ничего не перекодирует, идёт быстро.

Запуск:  .\.venv312\Scripts\python discourse_analysis.py
"""
import os
import re
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:  # вывод Cyrillic/× в консоль при запуске через PowerShell (иначе cp1251)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
N_REDUCED = 8          # целевое число укрупнённых тем (BERTopic учитывает -1)
MIN_CLASS_FOR_MLP = 8  # классы мельче — исключаем из supervised-слоя


def hr(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def clean_snippet(text, n=400):
    s = re.sub(r"\s+", " ", str(text)).strip()
    return s[:n] + ("…" if len(s) > n else "")


# === ЗАГРУЗКА ПЕРЕИСПОЛЬЗУЕМЫХ АРТЕФАКТОВ ===
hr("ЗАГРУЗКА ЭМБЕДДИНГОВ И ДАННЫХ")
csv_path = os.path.join(OUT, "cleaned_policies.csv")
emb_path = os.path.join(OUT, "embeddings_bge.npy")
if not (os.path.exists(csv_path) and os.path.exists(emb_path)):
    print("Нет outputs/cleaned_policies.csv или embeddings_bge.npy — сначала запустите main.py")
    sys.exit(1)

df = pd.read_csv(csv_path)
emb = np.load(emb_path).astype(np.float32)
texts = df["policy_text"].astype(str).tolist()
assert len(texts) == emb.shape[0], "Размерность эмбеддингов не совпадает с числом текстов"
print(f"Документов: {len(df)}, эмбеддинги bge: {emb.shape}")


# === ФАЗА A: BERTOPIC + ДОСЬЕ КЛАСТЕРОВ ===
hr("ФАЗА A: КЛАСТЕРЫ ДИСКУРСА (BERTopic)")
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer


def build_bertopic(min_topic_size):
    umap_model = UMAP(n_components=5, n_neighbors=10, min_dist=0.0,
                      metric="cosine", random_state=RANDOM_STATE)
    hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=3,
                            metric="euclidean", prediction_data=True)
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    return BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer, min_topic_size=min_topic_size,
                    calculate_probabilities=False, verbose=False)


topic_model = build_bertopic(5)
topics_full = topic_model.fit_transform(texts, embeddings=emb)[0]
if len([t for t in set(topics_full) if t != -1]) == 0:
    print("Всё ушло в шум — повтор с min_topic_size=3")
    topic_model = build_bertopic(3)
    topics_full = topic_model.fit_transform(texts, embeddings=emb)[0]

df["topic"] = topics_full
n_full = len([t for t in set(topics_full) if t != -1])
noise_pct = 100.0 * np.mean(np.array(topics_full) == -1)
print(f"Сырых кластеров: {n_full}; шум (-1): {noise_pct:.1f}%")


def topic_dossier(model, topics, title):
    """Собрать markdown-досье по кластерам модели."""
    lines = [f"### {title}\n"]
    arr = np.array(topics)
    for tid in sorted(set(topics)):
        if tid == -1:
            continue
        words = [w for w, _ in (model.get_topic(tid) or [])][:15]
        n = int((arr == tid).sum())
        sub = df[arr == tid]
        disc = ", ".join(f"{k}×{v}" for k, v in
                         Counter(sub["category"]).most_common())
        try:
            reps = model.get_representative_docs(tid)[:3]
        except Exception:
            reps = sub["policy_text"].head(3).tolist()
        lines.append(f"#### Кластер {tid} (n={n})  —  **[НАЗВАНИЕ ВРУЧНУЮ]**")
        lines.append(f"- Ключевые слова: {', '.join(words)}")
        lines.append(f"- По категориям: {disc}")
        lines.append("- Репрезентативные тексты:")
        for r in reps:
            lines.append(f"  - «{clean_snippet(r)}»")
        lines.append("")
    return "\n".join(lines)


dossier_full = topic_dossier(topic_model, topics_full, f"14 сырых кластеров (n_topics={n_full})")

# Укрупнение
hr("ФАЗА A.2: УКРУПНЕНИЕ КЛАСТЕРОВ")
try:
    topic_model.reduce_topics(texts, nr_topics=N_REDUCED)
    topics_reduced = list(topic_model.topics_)
except Exception as e:
    print(f"reduce_topics не отработал ({e}); использую сырые кластеры как укрупнённые.")
    topics_reduced = list(topics_full)

df["reduced_topic"] = topics_reduced
n_red = len([t for t in set(topics_reduced) if t != -1])
print(f"Укрупнённых кластеров: {n_red}")
dossier_red = topic_dossier(topic_model, topics_reduced,
                            f"Укрупнённые кластеры (n_topics={n_red}) — рабочая основа типологии")

with open(os.path.join(OUT, "discourse_clusters.md"), "w", encoding="utf-8") as f:
    f.write("# Досье дискурсивных кластеров AI-политик\n\n"
            "Прочитайте ключевые слова и репрезентативные тексты, затем дайте каждому "
            "укрупнённому кластеру название и сгруппируйте в типологию диссертации.\n\n"
            + dossier_red + "\n---\n\n" + dossier_full)
df[["Course", "Discipline", "category", "topic", "reduced_topic"]].to_csv(
    os.path.join(OUT, "cluster_assignments.csv"), index=False, encoding="utf-8")
print("Сохранено: outputs/discourse_clusters.md, outputs/cluster_assignments.csv")


# === ФАЗА C: SUPERVISED НА ДИСКУРС-МЕТКАХ ===
hr("ФАЗА C: MLP + SHAP НА ДИСКУРС-КЛАСТЕРАХ")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu",
                         solver="adam", alpha=0.001, batch_size=32,
                         learning_rate_init=0.001, max_iter=500, early_stopping=True,
                         validation_fraction=0.15, n_iter_no_change=20,
                         random_state=RANDOM_STATE, verbose=False)


# Исключаем шум и слишком мелкие кластеры
mask = df["reduced_topic"] != -1
counts = Counter(df.loc[mask, "reduced_topic"])
keep = {t for t, c in counts.items() if c >= MIN_CLASS_FOR_MLP}
dropped = sorted(set(counts) - keep)
if dropped:
    print(f"Исключены из supervised мелкие кластеры (<{MIN_CLASS_FOR_MLP}): {dropped}")
sup = df[df["reduced_topic"].isin(keep)].reset_index(drop=True)
sup_emb = emb[df["reduced_topic"].isin(keep).values]
print(f"В supervised: {len(sup)} документов, классов: {sup['reduced_topic'].nunique()} "
      f"(шум {noise_pct:.1f}% исключён)")

le = LabelEncoder()
y = le.fit_transform(sup["reduced_topic"].values)
disc_names = [f"discourse_{c}" for c in le.classes_]

Xtr, Xte, ytr, yte = train_test_split(sup_emb, y, test_size=0.2, stratify=y,
                                      random_state=RANDOM_STATE)
mlp = make_mlp(); mlp.fit(Xtr, ytr)
ypred = mlp.predict(Xte)
acc = accuracy_score(yte, ypred)
majority = max(Counter(y).values()) / len(y)
print(f"\nDiscourse-MLP accuracy: {acc:.3f} (доля мажорного класса ~{majority:.3f})")
print("Рамка: высокая accuracy = кластеры лингвистически когерентны/разделимы; "
      "supervised-слой здесь — инструмент интерпретации, не предсказательное утверждение.")
print(classification_report(yte, ypred, target_names=disc_names, zero_division=0))

# Кривая обучения
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, label="train loss")
plt.xlabel("итерация"); plt.ylabel("train loss")
plt.title("Кривая обучения (дискурс-MLP)"); plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT, "discourse_learning_curve.png"), dpi=120)
plt.close()

cm = confusion_matrix(yte, ypred, normalize="true")
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=disc_names, yticklabels=disc_names)
plt.xlabel("Предсказано"); plt.ylabel("Истинно")
plt.title(f"Confusion matrix (дискурс), acc={acc:.2f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "discourse_confusion_matrix.png"), dpi=120)
plt.close()
print("Сохранено: discourse_learning_curve.png, discourse_confusion_matrix.png")


# SHAP по словам на дискурс-тип
hr("SHAP: РАЗЛИЧАЮЩИЕ СЛОВА НА ДИСКУРС-ТИП")
shap_words = {}
acc_tf = float("nan")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import shap

    txt = sup["policy_text"].astype(str).tolist()
    txt_tr, txt_te, ys_tr, ys_te = train_test_split(
        txt, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    tfidf = TfidfVectorizer(max_features=300, min_df=2, stop_words="english")
    Xtr_tf = tfidf.fit_transform(txt_tr).toarray()
    Xte_tf = tfidf.transform(txt_te).toarray()
    feats = np.array(tfidf.get_feature_names_out())

    mlp_tf = make_mlp(); mlp_tf.fit(Xtr_tf, ys_tr)
    acc_tf = accuracy_score(ys_te, mlp_tf.predict(Xte_tf))
    print(f"Accuracy TF-IDF MLP (для SHAP): {acc_tf:.3f}")

    n_te = min(len(Xte_tf), 25)
    rng = np.random.RandomState(RANDOM_STATE)
    bg = Xtr_tf[rng.choice(len(Xtr_tf), size=min(30, len(Xtr_tf)), replace=False)]
    explainer = shap.KernelExplainer(mlp_tf.predict_proba, bg)
    sv = explainer.shap_values(Xte_tf[:n_te], nsamples=100)
    sv_by_class = sv if isinstance(sv, list) else [sv[:, :, c] for c in range(sv.shape[2])]

    plt.figure()
    shap.summary_plot(sv_by_class, Xte_tf[:n_te], feature_names=feats,
                      class_names=disc_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "discourse_shap_summary.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("Сохранено: outputs/discourse_shap_summary.png")

    present = sorted(set(ys_tr) | set(ys_te))
    print("\nТоп-10 различающих слов на дискурс-кластер:")
    for ci in range(len(sv_by_class)):
        cname = disc_names[ci]
        mean_abs = np.abs(sv_by_class[ci]).mean(axis=0)
        top = list(feats[np.argsort(mean_abs)[::-1][:10]])
        shap_words[cname] = top
        print(f"  {cname}: {', '.join(top)}")
except Exception as e:
    print(f"SHAP не отработал: {e}")


# === ФАЗА C.2: КРОССТАБ ДИСКУРС x ДИСЦИПЛИНА + ОТЧЁТ ===
hr("СИНТЕЗ: ДИСКУРС × ДИСЦИПЛИНА И ОТЧЁТ")
ct = pd.crosstab(df["reduced_topic"], df["category"])
print("Кросстаб (укрупнённый дискурс-кластер × дисциплинарная категория):")
print(ct.to_string())


def md_table(dft):
    cols = list(dft.columns)
    label = dft.index.name or "cluster"
    rows = ["| " + label + " | " + " | ".join(map(str, cols)) + " |",
            "|" + "---|" * (len(cols) + 1)]
    for idx, r in dft.iterrows():
        rows.append(f"| {idx} | " + " | ".join(str(int(v)) for v in r) + " |")
    return "\n".join(rows)


def md_shap():
    if not shap_words:
        return "_SHAP не выполнен._"
    rows = ["| Дискурс-кластер | Топ-10 различающих слов |", "|---|---|"]
    for c, ws in shap_words.items():
        rows.append(f"| {c} | {', '.join(ws)} |")
    return "\n".join(rows)


report = f"""# Дискурс-анализ AI-политик университетских силлабусов

> Главный результат — **дискурсивные типы**, найденные снизу вверх (без предзаданных
> категорий). Дисциплина используется как вторичная ось. Дисциплинарная классификация
> из `main.py`/`analysis_summary.md` понижена до побочного результата.

## 1. Данные
- {len(df)} документов (корпус Lance Eaton), эмбеддинги `BAAI/bge-base-en-v1.5`.

## 2. Кластеры дискурса (BERTopic, unsupervised)
- Сырых кластеров: {n_full}; шум (-1): {noise_pct:.1f}%.
- Укрупнённых кластеров (рабочая типология): {n_red}.
- Детальное досье с ключевыми словами и репрезентативными текстами: `discourse_clusters.md`.
- Интерактивная карта: `umap_by_topic.html`.

## 3. Лингвистическая разделимость (дискурс-MLP)
- Accuracy предсказания дискурс-кластера по эмбеддингу: **{acc:.3f}** (мажорный класс ~{majority:.3f}).
- Трактовка: это мера **внутренней когерентности** кластеров, а не предсказательное
  утверждение (метки выведены из тех же эмбеддингов). Файлы: `discourse_confusion_matrix.png`,
  `discourse_learning_curve.png`.

## 4. Что лингвистически отличает каждый дискурс-тип (SHAP)
Accuracy вспомогательного TF-IDF MLP: {acc_tf:.3f}

{md_shap()}

## 5. Дискурс × дисциплина (вторичная ось)
Связаны ли дискурсивные типы с дисциплинами:

{md_table(ct)}

## 6. Интерпретация кластеров [ЗАПОЛНИТЬ ВРУЧНУЮ]
Названия укрупнённых кластеров (по `discourse_clusters.md`):
- discourse_0 = ...
- discourse_1 = ...
(и т.д.)

## 7. Связь с типологией диссертации [ЗАПОЛНИТЬ ВРУЧНУЮ]
Соотнесение найденных дискурс-типов с осями: этический минимализм / гуманистический /
прагматико-технократический.

## 8. Выводы [ЗАПОЛНИТЬ ВРУЧНУЮ]
"""
with open(os.path.join(OUT, "discourse_summary.md"), "w", encoding="utf-8") as f:
    f.write(report)
print("Сохранено: outputs/discourse_summary.md")

# === ФАЗА D: KMEANS-ТИПОЛОГИЯ (полная, без шума, сбалансированная) ===
hr("ФАЗА D: KMEANS-ТИПОЛОГИЯ")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Обоснование выбора k: silhouette по эмбеддингам
print("Silhouette по k (выбор числа дискурс-типов):")
sil = {}
for k in range(3, 8):
    km_k = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(emb)
    sil[k] = silhouette_score(emb, km_k.labels_, metric="cosine")
    print(f"  k={k}: silhouette={sil[k]:.3f}")
K = 5  # типология из 5 дискурс-типов (рекомендованная гранулярность)
print(f"Берём k={K} (каждый документ -> один дискурс-тип, без шума).")

km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(emb)
df["kmeans"] = km.labels_

# Характерные слова на кластер (по среднему TF-IDF) + репрезентативные тексты (центроид)
tf_all = TfidfVectorizer(max_features=2000, min_df=2, stop_words="english")
M = tf_all.fit_transform(texts)
fnames = np.array(tf_all.get_feature_names_out())
centroids_norm = np.zeros((K, emb.shape[1]), dtype=np.float32)
for c in range(K):
    v = emb[km.labels_ == c].mean(axis=0)
    centroids_norm[c] = v / (np.linalg.norm(v) + 1e-9)

km_lines = ["\n---\n\n### KMeans-типология (k=%d, без шума, полная)\n" % K]
for c in range(K):
    idx = np.where(km.labels_ == c)[0]
    mean_tfidf = np.asarray(M[idx].mean(axis=0)).ravel()
    topw = ", ".join(fnames[np.argsort(mean_tfidf)[::-1][:15]])
    sims = emb[idx] @ centroids_norm[c]
    reps = idx[np.argsort(sims)[::-1][:3]]
    disc = ", ".join(f"{k2}×{v2}" for k2, v2 in
                     Counter(df.iloc[idx]["category"]).most_common())
    km_lines.append(f"#### KMeans-кластер {c} (n={len(idx)})  —  **[НАЗВАНИЕ ВРУЧНУЮ]**")
    km_lines.append(f"- Характерные слова (TF-IDF): {topw}")
    km_lines.append(f"- По категориям: {disc}")
    km_lines.append("- Репрезентативные тексты:")
    for r in reps:
        km_lines.append(f"  - «{clean_snippet(texts[r])}»")
    km_lines.append("")
with open(os.path.join(OUT, "discourse_clusters.md"), "a", encoding="utf-8") as f:
    f.write("\n".join(km_lines))

# MLP + SHAP на KMeans-метках
yk = km.labels_
kmn = [f"km_{c}" for c in range(K)]
Xtr, Xte, ytr, yte = train_test_split(emb, yk, test_size=0.2, stratify=yk,
                                      random_state=RANDOM_STATE)
mlp_k = make_mlp(); mlp_k.fit(Xtr, ytr)
acc_k = accuracy_score(yte, mlp_k.predict(Xte))
maj_k = max(Counter(yk).values()) / len(yk)
print(f"\nKMeans-MLP accuracy: {acc_k:.3f} (мажорный класс ~{maj_k:.3f})")
print(classification_report(yte, mlp_k.predict(Xte), target_names=kmn, zero_division=0))

cmk = confusion_matrix(yte, mlp_k.predict(Xte), normalize="true")
plt.figure(figsize=(7, 6))
sns.heatmap(cmk, annot=True, fmt=".2f", cmap="Greens", xticklabels=kmn, yticklabels=kmn)
plt.xlabel("Предсказано"); plt.ylabel("Истинно")
plt.title(f"KMeans confusion (норм.), acc={acc_k:.2f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "kmeans_confusion_matrix.png"), dpi=120)
plt.close()

km_shap = {}
acc_ktf = float("nan")
try:
    import shap
    tfk = TfidfVectorizer(max_features=300, min_df=2, stop_words="english")
    txt_tr, txt_te, ys_tr, ys_te = train_test_split(
        texts, yk, test_size=0.2, stratify=yk, random_state=RANDOM_STATE)
    Xtr_tf = tfk.fit_transform(txt_tr).toarray()
    Xte_tf = tfk.transform(txt_te).toarray()
    fk = np.array(tfk.get_feature_names_out())
    mtf = make_mlp(); mtf.fit(Xtr_tf, ys_tr)
    acc_ktf = accuracy_score(ys_te, mtf.predict(Xte_tf))
    print(f"Accuracy TF-IDF MLP (KMeans, для SHAP): {acc_ktf:.3f}")
    rng = np.random.RandomState(RANDOM_STATE)
    bg = Xtr_tf[rng.choice(len(Xtr_tf), size=min(30, len(Xtr_tf)), replace=False)]
    ex = shap.KernelExplainer(mtf.predict_proba, bg)
    n_te = min(len(Xte_tf), 25)
    sv = ex.shap_values(Xte_tf[:n_te], nsamples=100)
    svc = sv if isinstance(sv, list) else [sv[:, :, c] for c in range(sv.shape[2])]
    plt.figure()
    shap.summary_plot(svc, Xte_tf[:n_te], feature_names=fk, class_names=kmn,
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "kmeans_shap_summary.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("Топ-10 различающих слов на KMeans-тип:")
    for ci in range(len(svc)):
        top = list(fk[np.argsort(np.abs(svc[ci]).mean(axis=0))[::-1][:10]])
        km_shap[kmn[ci]] = top
        print(f"  {kmn[ci]}: {', '.join(top)}")
except Exception as e:
    print(f"SHAP (KMeans) не отработал: {e}")

ctk = pd.crosstab(df["kmeans"], df["category"])
print("\nКросстаб (KMeans-тип × дисциплина):")
print(ctk.to_string())

# Дописать KMeans-секцию в отчёт и пересохранить assignments с колонкой kmeans
def md_kshap():
    if not km_shap:
        return "_SHAP не выполнен._"
    rows = ["| KMeans-тип | Топ-10 различающих слов |", "|---|---|"]
    for c, ws in km_shap.items():
        rows.append(f"| {c} | {', '.join(ws)} |")
    return "\n".join(rows)

km_report = f"""

---

## 9. Альтернатива: KMeans-типология (полная, сбалансированная)

В отличие от BERTopic/HDBSCAN (плотностный, оставляет {noise_pct:.0f}% шума и одно
доминирующее ядро), KMeans приписывает **каждый** документ к одному из k дискурс-типов.

- Выбор k по silhouette (cosine): {', '.join(f'k={k}:{v:.3f}' for k, v in sil.items())}.
- Принят **k={K}**. Размеры классов: {', '.join(f'km_{int(c)}={int(n)}' for c, n in sorted(Counter(km.labels_).items()))}.
- KMeans-MLP accuracy: **{acc_k:.3f}** (мажорный класс ~{maj_k:.3f}); TF-IDF MLP: {acc_ktf:.3f}.
- Файлы: `kmeans_confusion_matrix.png`, `kmeans_shap_summary.png`; досье — в `discourse_clusters.md`.

Различающие слова на KMeans-тип (SHAP):

{md_kshap()}

Кросстаб KMeans-тип × дисциплина:

{md_table(ctk.rename_axis('kmeans'))}

### Названия KMeans-типов [ЗАПОЛНИТЬ ВРУЧНУЮ]
{chr(10).join(f'- km_{c} = ...' for c in range(K))}
"""
with open(os.path.join(OUT, "discourse_summary.md"), "a", encoding="utf-8") as f:
    f.write(km_report)
df[["Course", "Discipline", "category", "topic", "reduced_topic", "kmeans"]].to_csv(
    os.path.join(OUT, "cluster_assignments.csv"), index=False, encoding="utf-8")
print("Дописано в discourse_summary.md и discourse_clusters.md; assignments обновлены.")

print("\n" + "=" * 40)
print("=== ДИСКУРС-АНАЛИЗ ЗАВЕРШЁН ===")
print(f"BERTopic: сырых {n_full} | укрупнённых {n_red} | шум {noise_pct:.1f}% | MLP {acc*100:.1f}%")
print(f"KMeans:   k={K} (без шума) | MLP {acc_k*100:.1f}%")
print("Читать: outputs/discourse_clusters.md -> назвать кластеры -> "
      "заполнить поля в discourse_summary.md")
print("=" * 40)
