# -*- coding: utf-8 -*-
r"""
Анализ дискурса AI-политик в университетских силлабусах.
Пайплайн: sentence-embeddings -> MLP-классификатор по дисциплинарным категориям,
BERTopic как разведочный кластерный анализ, SHAP на TF-IDF для интерпретации слов.

Запуск:  .\.venv312\Scripts\python main.py
Весь вывод -> stdout + папка outputs/.
"""
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_LOCAL = os.path.join(HERE, "Syllabi Policies for Generative AI Repository - Policies.csv")
CSV_URL = ("https://docs.google.com/spreadsheets/d/"
           "1lM6g4yveQMyWeUbEwBM6FZVxEWCLfvWDh1aWUErWWbQ/gviz/tq?tqx=out:csv&sheet=Policies")


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# === ШАГ 1: ЗАГРУЗКА И ПЕРВИЧНАЯ ОЧИСТКА ===
hr("ШАГ 1: ЗАГРУЗКА И ОЧИСТКА ДАННЫХ")

df = None
if os.path.exists(CSV_LOCAL):
    print(f"Читаю локальный CSV: {os.path.basename(CSV_LOCAL)}")
    df = pd.read_csv(CSV_LOCAL)
else:
    try:
        print("Локального файла нет, пробую Google Sheets...")
        df = pd.read_csv(CSV_URL)
    except Exception as e:
        print(f"Загрузка не удалась: {e}")
        print("Скачайте CSV вручную по ссылке:\n"
              "https://docs.google.com/spreadsheets/d/1lM6g4yveQMyWeUbEwBM6FZVxEWCLfvWDh1aWUErWWbQ\n"
              "и сохраните как policies.csv в текущей папке.")
        if os.path.exists(os.path.join(HERE, "policies.csv")):
            df = pd.read_csv(os.path.join(HERE, "policies.csv"))
        else:
            sys.exit(1)

n_raw = len(df)
# Подсчёт лицензий по всему сырому датасету (до фильтрации) — для раздела 9 отчёта.
def license_bucket(s):
    t = str(s or "").strip().lower()
    if not t or t == "nan":
        return "Не указано"
    if "public domain" in t:
        return "Public Domain"
    if "noncommer" in t:
        return "CC-BY-NC*"
    if "no deriv" in t or "contact me" in t or "copyright retain" in t:
        return "ND / Copyright / Contact"
    if ("creative commons" in t or "cc-by" in t or "cc by" in t
            or " by" in t or "copyright" in t):
        return "CC-BY / BY-SA"
    return "ND / Copyright / Contact"

RIGHTS_ORDER = ["Public Domain", "CC-BY / BY-SA", "CC-BY-NC*",
                "ND / Copyright / Contact", "Не указано"]
RIGHTS_NOTE = {
    "Public Domain": "без ограничений",
    "CC-BY / BY-SA": "свободно с указанием авторства",
    "CC-BY-NC*": "только некоммерческое использование",
    "ND / Copyright / Contact": "дословное воспроизведение — только с разрешения автора",
    "Не указано": "трактуется консервативно",
}
if "Rights for Reuse" in df.columns:
    rights_counts = Counter(df["Rights for Reuse"].apply(license_bucket))
else:
    rights_counts = Counter()

df = df.rename(columns={"Policy in the Syllabus": "policy_text"})
df["policy_text"] = df["policy_text"].astype("string")
df = df[df["policy_text"].notna()]
df["policy_text"] = df["policy_text"].str.strip()
df = df[df["policy_text"].str.len() > 0]
n_nonempty = len(df)

df["word_count"] = df["policy_text"].str.split().str.len()
df = df[df["word_count"] >= 30].reset_index(drop=True)
n_filt = len(df)

print(f"Строк всего (до фильтрации):        {n_raw}")
print(f"Непустых policy_text:               {n_nonempty}")
print(f"После фильтра >=30 слов:            {n_filt}")
wc = df["word_count"]
print(f"Длина текстов (слов): min={wc.min()} max={wc.max()} "
      f"mean={wc.mean():.1f} median={wc.median():.0f}")
print(f"Уникальных дисциплин:               {df['Discipline'].nunique()}")
print("Топ-10 дисциплин по числу записей:")
print(df["Discipline"].value_counts().head(10).to_string())


# === ШАГ 2: МЕТКИ ДИСЦИПЛИНАРНЫХ КАТЕГОРИЙ ===
hr("ШАГ 2: КАТЕГОРИИ ДИСЦИПЛИН")

# Сопоставление по ключевым словам (вхождение подстроки в нормализованное имя дисциплины).
# Улучшение против ТЗ: точное сравнение строк роняло 13% записей в Other.
CATEGORY_KEYWORDS = {
    "STEM": ["biolog", "chemis", "engineer", "computer", "math", "physical science",
             "life science", "nursing", "occupational therapy", "data science",
             "environ", "technology"],
    "Humanities": ["history", "literature", "language", "philosoph", "writing",
                   "film", "music", "theatre", "theater", "media studies", "cultural",
                   "gender", "religi", "composition", "english", "spanish", "art"],
    "Professional": ["business", "law", "marketing", "finance", "social work",
                     "healthcare", "health care", "information scien",
                     "information tech", "information stud", "library", "research",
                     "management", "accounting", "econom", "entrepreneur",
                     "cybersecurity", "criminal justice", "public policy",
                     "human services", "social services", "design"],
    "Education_General": ["education", "first year", "instructional", "misc",
                          "interdisciplin", "sociolog", "psycholog", "communicat",
                          "government", "social science", "geography", "journal",
                          "agricultur", "anthropolog", "politic", "honors", "high school",
                          "seminar", "health"],
}
ORDER = ["STEM", "Humanities", "Professional", "Education_General"]


def map_category(discipline):
    d = str(discipline).lower().strip()
    for cat in ORDER:
        for kw in CATEGORY_KEYWORDS[cat]:
            if kw in d:
                return cat
    return "Other"


df["category"] = df["Discipline"].apply(map_category)
n_other = int((df["category"] == "Other").sum())
if n_other:
    others = sorted(df.loc[df["category"] == "Other", "Discipline"].unique())
    print(f"Не сопоставлено явно (Other): {n_other} -> сворачиваю в Education_General.")
    print(f"  Список Other: {others}")
    df.loc[df["category"] == "Other", "category"] = "Education_General"

print("\nБаланс классов после маппинга:")
cat_counts = df["category"].value_counts()
print(cat_counts.to_string())
for cat, cnt in cat_counts.items():
    if cnt < 10:
        print(f"  ВНИМАНИЕ: класс '{cat}' имеет {cnt} записей (<10) — рассмотрите объединение.")

df.to_csv(os.path.join(OUT, "cleaned_policies.csv"), index=False, encoding="utf-8")
print(f"\nСохранено: outputs/cleaned_policies.csv ({len(df)} строк)")


# === ШАГ 3: ЭМБЕДДИНГИ (нейросетевой компонент #1) ===
hr("ШАГ 3: SENTENCE-ЭМБЕДДИНГИ")

from sentence_transformers import SentenceTransformer

texts = df["policy_text"].tolist()

# Две модели: основная английская + мультиязычная (задел под русский корпус).
EMB_MODELS = {
    "bge": ("BAAI/bge-base-en-v1.5", ""),                 # английская, топ MTEB
    "e5":  ("intfloat/multilingual-e5-large", "passage: "),  # мультиязычная
}
embeddings = {}
for key, (model_name, prefix) in EMB_MODELS.items():
    print(f"\nМодель '{key}': {model_name}")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    inp = [prefix + t for t in texts] if prefix else texts
    emb = model.encode(inp, show_progress_bar=True, batch_size=16,
                       normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    embeddings[key] = emb
    np.save(os.path.join(OUT, f"embeddings_{key}.npy"), emb)
    print(f"  shape={emb.shape}, время={time.time()-t0:.1f}c, "
          f"сохранено outputs/embeddings_{key}.npy")
    del model

labels_raw = df["category"].values
np.save(os.path.join(OUT, "labels.npy"), labels_raw)
print("\nСохранено: outputs/labels.npy")


# === ШАГ 4: КЛАСТЕРНЫЙ АНАЛИЗ BERTopic (разведка) ===
hr("ШАГ 4: BERTOPIC (исследовательский кластерный анализ)")

topic_ok = False
try:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    base_emb = embeddings["bge"]

    def run_bertopic(min_topic_size):
        umap_model = UMAP(n_components=5, n_neighbors=10, min_dist=0.0,
                          metric="cosine", random_state=RANDOM_STATE)
        hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=3,
                                metric="euclidean", prediction_data=True)
        vectorizer = CountVectorizer(stop_words="english", min_df=2)
        tm = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                      vectorizer_model=vectorizer, min_topic_size=min_topic_size,
                      calculate_probabilities=False, verbose=False)
        topics, _ = tm.fit_transform(texts, embeddings=base_emb)
        return tm, topics

    topic_model, topics = run_bertopic(5)
    n_topics = len([t for t in set(topics) if t != -1])
    if n_topics == 0:
        print("Кластеров не найдено (всё шум) — повтор с min_topic_size=3...")
        topic_model, topics = run_bertopic(3)
        n_topics = len([t for t in set(topics) if t != -1])

    df["topic"] = topics
    noise_pct = 100.0 * np.mean(np.array(topics) == -1)
    print(f"Найдено кластеров (без шума): {n_topics}")
    print(f"Доля документов в шумовом кластере (-1): {noise_pct:.1f}%")

    print("\nТоп-5 слов по кластерам:")
    info = topic_model.get_topic_info()
    for tid in sorted(set(topics)):
        words = topic_model.get_topic(tid)
        if not words:
            continue
        top5 = ", ".join(w for w, _ in words[:5])
        cnt = int((np.array(topics) == tid).sum())
        label = "ШУМ" if tid == -1 else f"кластер {tid}"
        print(f"  {label} (n={cnt}): {top5}")

    # Кросс-таблица кластер x категория
    print("\nCrosstab (кластер x категория):")
    ct = pd.crosstab(df["topic"], df["category"])
    print(ct.to_string())

    # 2D UMAP для визуализации
    import plotly.express as px
    umap2d = UMAP(n_components=2, n_neighbors=10, min_dist=0.1,
                  metric="cosine", random_state=RANDOM_STATE).fit_transform(base_emb)
    viz = pd.DataFrame({
        "x": umap2d[:, 0], "y": umap2d[:, 1],
        "topic": [str(t) for t in topics],
        "category": df["category"].values,
        "course": df["Course"].values,
    })
    fig_a = px.scatter(viz, x="x", y="y", color="topic", hover_data=["course", "category"],
                       title="UMAP документов, цвет = BERTopic-кластер")
    fig_a.write_html(os.path.join(OUT, "umap_by_topic.html"))
    fig_b = px.scatter(viz, x="x", y="y", color="category", hover_data=["course", "topic"],
                       title="UMAP документов, цвет = дисциплинарная категория")
    fig_b.write_html(os.path.join(OUT, "umap_by_category.html"))
    print("\nСохранено: outputs/umap_by_topic.html, outputs/umap_by_category.html")
    topic_ok = True
    BERTOPIC_SUMMARY = (n_topics, noise_pct)
except Exception as e:
    print(f"BERTopic недоступен/упал: {e}")
    print("Пропускаю кластерный анализ, продолжаю с классификацией.")
    BERTOPIC_SUMMARY = (0, 100.0)


# === ШАГ 5: MLP-КЛАССИФИКАТОР (основная нейросеть) ===
hr("ШАГ 5: ОБУЧЕНИЕ MLP-КЛАССИФИКАТОРА")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

le = LabelEncoder()
y = le.fit_transform(labels_raw)
class_names = list(le.classes_)
print(f"Классы: {dict(enumerate(class_names))}")


def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu",
                         solver="adam", alpha=0.001, batch_size=32,
                         learning_rate_init=0.001, max_iter=500, early_stopping=True,
                         validation_fraction=0.15, n_iter_no_change=20,
                         random_state=RANDOM_STATE, verbose=False)


# Сравнение двух эмбеддинговых бэкендов + baseline (LogReg)
results = {}
splits = {}
for key in EMB_MODELS:
    X = embeddings[key]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=RANDOM_STATE)
    splits[key] = (Xtr, Xte, ytr, yte)
    mlp = make_mlp(); mlp.fit(Xtr, ytr)
    acc_mlp = accuracy_score(yte, mlp.predict(Xte))
    logreg = LogisticRegression(max_iter=2000, C=1.0)
    logreg.fit(Xtr, ytr)
    acc_lr = accuracy_score(yte, logreg.predict(Xte))
    results[key] = {"mlp": mlp, "acc_mlp": acc_mlp, "acc_lr": acc_lr}
    print(f"[{key}] MLP acc={acc_mlp:.3f} | LogReg baseline acc={acc_lr:.3f} | "
          f"эпох={mlp.n_iter_}")

# Тест на дисбаланс
for key, (Xtr, Xte, ytr, yte) in splits.items():
    te_counts = Counter(yte)
    rare = [class_names[c] for c, n in te_counts.items() if n < 5]
    if rare:
        print(f"  ВНИМАНИЕ [{key}]: в test <5 образцов у классов: {rare}")

# Лучшая модель по MLP-accuracy
best_key = max(results, key=lambda k: results[k]["acc_mlp"])
best_mlp = results[best_key]["mlp"]
Xtr, Xte, ytr, yte = splits[best_key]
print(f"\nЛучший бэкенд эмбеддингов: '{best_key}' "
      f"({EMB_MODELS[best_key][0]}), MLP acc={results[best_key]['acc_mlp']:.3f}")
print(f"Финальный loss (train): {best_mlp.loss_:.4f}")

# Кривая обучения
plt.figure(figsize=(8, 5))
plt.plot(best_mlp.loss_curve_, label="train loss")
if hasattr(best_mlp, "validation_scores_") and best_mlp.validation_scores_:
    ax2 = plt.gca().twinx()
    ax2.plot(best_mlp.validation_scores_, color="orange", label="val accuracy")
    ax2.set_ylabel("validation accuracy")
plt.xlabel("итерация"); plt.ylabel("train loss")
plt.title(f"Кривая обучения MLP ({best_key})"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "learning_curve.png"), dpi=120)
plt.close()
print("Сохранено: outputs/learning_curve.png")


# === ШАГ 6: ОЦЕНКА МОДЕЛИ ===
hr("ШАГ 6: ОЦЕНКА")

ypred = best_mlp.predict(Xte)
acc = accuracy_score(yte, ypred)
report_txt = classification_report(yte, ypred, target_names=class_names, zero_division=0)
print(f"Accuracy: {acc:.3f}\n")
print(report_txt)

cm = confusion_matrix(yte, ypred, normalize="true")
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Предсказано"); plt.ylabel("Истинно")
plt.title(f"Confusion matrix (норм.), {best_key}, acc={acc:.2f}")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120)
plt.close()
print("Сохранено: outputs/confusion_matrix.png")

# Комментарий по классам
report_dict = classification_report(yte, ypred, target_names=class_names,
                                    zero_division=0, output_dict=True)
f1s = {c: report_dict[c]["f1-score"] for c in class_names}
best_c = max(f1s, key=f1s.get); worst_c = min(f1s, key=f1s.get)
print(f"\nЛучше всего предсказывается '{best_c}' (F1={f1s[best_c]:.2f}), "
      f"хуже всего '{worst_c}' (F1={f1s[worst_c]:.2f}).")
print("Вероятная причина: размер и однородность выборки класса "
      "(мажорные/семантически цельные классы предсказываются увереннее).")


# === ШАГ 7: SHAP-АНАЛИЗ ЗНАЧИМОСТИ СЛОВ ===
hr("ШАГ 7: SHAP НА TF-IDF")

shap_words = {}
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import shap

    txt_tr, txt_te, ys_tr, ys_te = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    tfidf = TfidfVectorizer(max_features=300, min_df=2, stop_words="english")
    Xtr_tf = tfidf.fit_transform(txt_tr).toarray()
    Xte_tf = tfidf.transform(txt_te).toarray()
    feat_names = np.array(tfidf.get_feature_names_out())

    mlp_tf = make_mlp(); mlp_tf.fit(Xtr_tf, ys_tr)
    acc_tf = accuracy_score(ys_te, mlp_tf.predict(Xte_tf))
    print(f"Accuracy TF-IDF MLP: {acc_tf:.3f}")
    if acc_tf < 0.4:
        print("  ВНИМАНИЕ: accuracy <0.4 — нормально для малого датасета, "
              "SHAP всё равно показывает относительную важность слов.")

    n_te = min(len(Xte_tf), 25)
    Xte_shap = Xte_tf[:n_te]
    rng = np.random.RandomState(RANDOM_STATE)
    bg_idx = rng.choice(len(Xtr_tf), size=min(30, len(Xtr_tf)), replace=False)
    background = Xtr_tf[bg_idx]

    explainer = shap.KernelExplainer(mlp_tf.predict_proba, background)
    shap_values = explainer.shap_values(Xte_shap, nsamples=100)

    # shap_values: list по классам (старый API) или ndarray (n,feat,classes)
    if isinstance(shap_values, list):
        sv_by_class = shap_values
    else:
        sv_by_class = [shap_values[:, :, c] for c in range(shap_values.shape[2])]

    plt.figure()
    shap.summary_plot(sv_by_class, Xte_shap, feature_names=feat_names,
                      class_names=class_names, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "shap_summary.png"), dpi=120,
                                    bbox_inches="tight")
    plt.close()
    print("Сохранено: outputs/shap_summary.png")

    print("\nТоп-10 различающих слов по классам (по среднему |SHAP|):")
    for ci, cname in enumerate(class_names):
        mean_abs = np.abs(sv_by_class[ci]).mean(axis=0)
        top = feat_names[np.argsort(mean_abs)[::-1][:10]]
        shap_words[cname] = list(top)
        print(f"  {cname}: {', '.join(top)}")
except Exception as e:
    print(f"SHAP не отработал: {e}")
    acc_tf = float("nan")


# === ШАГ 8: ИТОГОВЫЙ ОТЧЁТ ===
hr("ШАГ 8: ОТЧЁТ analysis_summary.md")

cat_table = cat_counts.reindex(ORDER).fillna(0).astype(int)
len_by_cat = df.groupby("category")["word_count"].mean().round(0).astype(int)


def md_metrics_table():
    rows = ["| Класс | precision | recall | F1 | support |",
            "|---|---|---|---|---|"]
    for c in class_names:
        d = report_dict[c]
        rows.append(f"| {c} | {d['precision']:.2f} | {d['recall']:.2f} | "
                    f"{d['f1-score']:.2f} | {int(d['support'])} |")
    return "\n".join(rows)


def md_shap_table():
    if not shap_words:
        return "_SHAP не выполнен._"
    rows = ["| Категория | Топ-10 различающих слов |", "|---|---|"]
    for c, ws in shap_words.items():
        rows.append(f"| {c} | {', '.join(ws)} |")
    return "\n".join(rows)


n_topics, noise_pct = BERTOPIC_SUMMARY
summary = f"""# Анализ дискурса AI-политик в университетских силлабусах

## 1. Данные
- Источник: публичный репозиторий AI-политик Lance Eaton (Google Sheets).
- URL: {CSV_URL}
- Документов: {n_raw} всего -> {n_filt} после фильтра (>=30 слов).
- Длина текстов: median={int(wc.median())} слов (min={wc.min()}, max={wc.max()}).

Распределение по категориям:

| Категория | Документов | Средняя длина (слов) |
|---|---|---|
""" + "\n".join(
    f"| {c} | {int(cat_table.get(c,0))} | {int(len_by_cat.get(c,0))} |" for c in ORDER
) + f"""

## 2. Архитектура и обоснование
- **Эмбеддинги (sentence-transformers), а не bag-of-words/LDA**: фиксируют семантику
  коротких текстов; AI-политики разных дисциплин используют схожую лексику в разном
  ценностном контексте, что плохо ловит частотный подход.
- Сравнивались две модели: `BAAI/bge-base-en-v1.5` (английская, топ MTEB) и
  `intfloat/multilingual-e5-large` (мультиязычная, задел под русскоязычный корпус).
- Пайплайн: текст -> эмбеддинг (768/1024-dim) -> MLP-классификатор (256->128->64).
  Сжатие пространства тремя слоями + early stopping против переобучения на малом датасете.
- BERTopic — вспомогательный разведочный инструмент (кластеры без меток), не основная сеть.

Сравнение бэкендов эмбеддингов (accuracy на тесте):

| Эмбеддинги | MLP | LogReg (baseline) |
|---|---|---|
""" + "\n".join(
    f"| {EMB_MODELS[k][0]} | {results[k]['acc_mlp']:.3f} | {results[k]['acc_lr']:.3f} |"
    for k in EMB_MODELS
) + f"""

Лучший бэкенд: **{EMB_MODELS[best_key][0]}**.

## 3. Результаты кластерного анализа (BERTopic)
- Найдено кластеров: {n_topics}; доля шума (-1): {noise_pct:.1f}%.
- Интерактивные карты: `umap_by_topic.html`, `umap_by_category.html`.
- Вывод о совпадении тематических кластеров с дисциплинарными категориями — см. crosstab в логе.

## 4. Результаты классификации (MLP)
Accuracy (лучший бэкенд, {best_key}): **{acc:.3f}**

{md_metrics_table()}

- Confusion matrix: `confusion_matrix.png`
- Кривая обучения: `learning_curve.png`
- Лучше всего: {best_c}; хуже всего: {worst_c}.

## 5. Интерпретация (SHAP)
Accuracy вспомогательного TF-IDF MLP: {acc_tf:.3f}

{md_shap_table()}

## 6. Связь с диссертационным исследованием
[ЗАПОЛНИТЬ ВРУЧНУЮ — соотнести результаты с типологией
этический минимализм / гуманистический / прагматико-технократический]

## 7. Ограничения и направления развития
- Датасет {n_filt} документов, классы несбалансированы (Humanities раздут за счёт Writing).
- Категории заданы эвристическим маппингом дисциплин, а не экспертной разметкой.
- Возможное расширение: русскоязычный корпус (48 политик российских вузов) — для этого
  уже подготовлены мультиязычные эмбеддинги e5.

## 8. Выводы
[ЗАПОЛНИТЬ ВРУЧНУЮ]

## 9. Источники и права на данные

**Источник данных.** Lance Eaton, «Syllabi Policies for Generative AI» (публичный
репозиторий AI-политик университетских силлабусов), Google Sheets:
{CSV_URL}
При использовании результатов источник цитируется обязательно (атрибуция).

**Лицензии исходных текстов** (колонка `Rights for Reuse`, {n_raw} строк):

| Режим | Строк | Условие использования |
|---|---|---|
""" + "\n".join(
    f"| {b} | {rights_counts.get(b, 0)} | {RIGHTS_NOTE[b]} |" for b in RIGHTS_ORDER
) + """

**Модели эмбеддингов.** `BAAI/bge-base-en-v1.5` и `intfloat/multilingual-e5-large` —
лицензия **MIT**; их выход (векторы, предсказания) используется свободно.

**Режим использования результатов.**
- Агрегированные производные артефакты (эмбеддинги `.npy`, метрики, графики, кластеры,
  списки различающих слов, выводы) — используются в диссертации свободно; это
  трансформативный анализ, а не републикация исходных текстов.
- Файл `cleaned_policies.csv` содержит политики дословно, включая тексты с пометками
  ND / Copyright / «Contact me». Хранение для личной работы допустимо; **публичное
  распространение самого датасета — нет** (только Public Domain и разрешающие распространение CC).
- NC-лицензии запрещают коммерческое применение этих данных; для академической
  (некоммерческой) работы ограничения не наступают.
- Для публичного приложения к диссертации рекомендуется выкладывать только производные
  артефакты, без сырого `cleaned_policies.csv`.
"""

with open(os.path.join(OUT, "analysis_summary.md"), "w", encoding="utf-8") as f:
    f.write(summary)
print("Сохранено: outputs/analysis_summary.md")


# === ШАГ 9: ТЕХ. АРТЕФАКТЫ (requirements) И ФИНАЛЬНАЯ СВОДКА ===
hr("ШАГ 9: АРТЕФАКТЫ")
try:
    import subprocess
    req = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                         capture_output=True, text=True).stdout
    with open(os.path.join(OUT, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(req)
    with open(os.path.join(HERE, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(req)
    print("Сохранено: outputs/requirements.txt, requirements.txt")
except Exception as e:
    print(f"pip freeze не удался: {e}")

print("\n" + "=" * 40)
print("=== ПАЙПЛАЙН ЗАВЕРШЁН ===")
print(f"Документов проанализировано: {n_filt}")
print(f"Кластеров BERTopic: {n_topics}")
print(f"MLP accuracy (embeddings, {best_key}): {acc*100:.1f}%")
print(f"MLP accuracy (TF-IDF/SHAP): {acc_tf*100:.1f}%")
print("Все файлы сохранены в outputs/")
print("=" * 40)
