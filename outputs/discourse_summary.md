# Дискурс-анализ AI-политик университетских силлабусов

> Главный результат — **дискурсивные типы**, найденные снизу вверх (без предзаданных
> категорий). Дисциплина используется как вторичная ось. Дисциплинарная классификация
> из `main.py`/`analysis_summary.md` понижена до побочного результата.

## 1. Данные
- 208 документов (корпус Lance Eaton), эмбеддинги `BAAI/bge-base-en-v1.5`.

## 2. Кластеры дискурса (BERTopic, unsupervised)
- Сырых кластеров: 14; шум (-1): 23.6%.
- Укрупнённых кластеров (рабочая типология): 7.
- Детальное досье с ключевыми словами и репрезентативными текстами: `discourse_clusters.md`.
- Интерактивная карта: `umap_by_topic.html`.

## 3. Лингвистическая разделимость (дискурс-MLP)
- Accuracy предсказания дискурс-кластера по эмбеддингу: **0.897** (мажорный класс ~0.622).
- Трактовка: это мера **внутренней когерентности** кластеров, а не предсказательное
  утверждение (метки выведены из тех же эмбеддингов). Файлы: `discourse_confusion_matrix.png`,
  `discourse_learning_curve.png`.

## 4. Что лингвистически отличает каждый дискурс-тип (SHAP)
Accuracy вспомогательного TF-IDF MLP: 0.828

| Дискурс-кластер | Топ-10 различающих слов |
|---|---|
| discourse_0 | problem, generative, university, language, permitted, cheating, reflection, writing, critical, ai |
| discourse_1 | ai, generative, writing, students, training, question, project, review, written, ethical |
| discourse_2 | project, generative, academic, writing, author, tool, ai, don, honesty, university |
| discourse_3 | ai, critical, clear, chatgpt, description, prompt, skills, end, systems, generative |

## 5. Дискурс × дисциплина (вторичная ось)
Связаны ли дискурсивные типы с дисциплинами:

| reduced_topic | Education_General | Humanities | Professional | STEM |
|---|---|---|---|---|
| -1 | 18 | 11 | 12 | 8 |
| 0 | 18 | 37 | 22 | 12 |
| 1 | 5 | 6 | 7 | 2 |
| 2 | 5 | 9 | 1 | 4 |
| 3 | 2 | 6 | 5 | 2 |
| 4 | 1 | 2 | 1 | 2 |
| 5 | 2 | 0 | 1 | 2 |
| 6 | 1 | 4 | 0 | 0 |

## 6. Интерпретация кластеров [ЗАПОЛНИТЬ ВРУЧНУЮ]
Названия укрупнённых кластеров (по `discourse_clusters.md`):
- discourse_0 = ...
- discourse_1 = ...
(и т.д.)

## 7. Связь с типологией диссертации [ЗАПОЛНИТЬ ВРУЧНУЮ]
Соотнесение найденных дискурс-типов с осями: этический минимализм / гуманистический /
прагматико-технократический.

## 8. Выводы [ЗАПОЛНИТЬ ВРУЧНУЮ]


---

## 9. Альтернатива: KMeans-типология (полная, сбалансированная)

В отличие от BERTopic/HDBSCAN (плотностный, оставляет 24% шума и одно
доминирующее ядро), KMeans приписывает **каждый** документ к одному из k дискурс-типов.

- Выбор k по silhouette (cosine): k=3:0.101, k=4:0.090, k=5:0.082, k=6:0.058, k=7:0.069.
- Принят **k=5**. Размеры классов: km_0=25, km_1=41, km_2=61, km_3=54, km_4=27.
- KMeans-MLP accuracy: **0.929** (мажорный класс ~0.293); TF-IDF MLP: 0.714.
- Файлы: `kmeans_confusion_matrix.png`, `kmeans_shap_summary.png`; досье — в `discourse_clusters.md`.

Различающие слова на KMeans-тип (SHAP):

| KMeans-тип | Топ-10 различающих слов |
|---|---|
| km_0 | writing, tools, materials, code, ai, ethically, let, paper, discussion, generated |
| km_1 | generative, including, reflection, ai, remember, explore, making, university, good, source |
| km_2 | version, ai, chatgpt, paste, feedback, assignment, graded, write, biased, understanding |
| km_3 | ai, general, grade, using, models, topics, generative, learn, concepts, tools |
| km_4 | ai, dishonesty, ensure, complete, plagiarism, tools, instructor, university, academic, version |

Кросстаб KMeans-тип × дисциплина:

| kmeans | Education_General | Humanities | Professional | STEM |
|---|---|---|---|---|
| 0 | 4 | 15 | 4 | 2 |
| 1 | 12 | 8 | 13 | 8 |
| 2 | 12 | 22 | 16 | 11 |
| 3 | 16 | 17 | 14 | 7 |
| 4 | 8 | 13 | 2 | 4 |

### Названия KMeans-типов [ЗАПОЛНИТЬ ВРУЧНУЮ]
- km_0 = ...
- km_1 = ...
- km_2 = ...
- km_3 = ...
- km_4 = ...
