# 📮 Анализатор почтовых марок

Python-скрипт для анализа изображений почтовых марок с использованием OpenAI GPT-4.1 multimodal API. Извлекает визуальную информацию и дополняет её справочными данными из открытых источников.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)
![python-dotenv](https://img.shields.io/badge/python--dotenv-3776AB?logo=python&logoColor=white)
![JSON](https://img.shields.io/badge/JSON-000000?logo=json&logoColor=white)

---

## 📋 Описание

Этот проект позволяет анализировать изображения почтовых марок через OpenAI GPT-4.1, извлекая визуально доступную информацию (страна, номинал, год, сюжет и т.д.) и дополняя её справочным описанием на основе общедоступных знаний.

### Основные возможности

* ✅ Анализ изображений почтовых марок через OpenAI GPT-4.1
* ✅ Извлечение визуальной информации (страна, номинал, год, текст, цвета)
* ✅ Справочная информация о марке и историческом контексте
* ✅ Сохранение результатов в формате JSON
* ✅ Генерация голосового описания марки (опционально)
* ✅ Детерминированные результаты (temperature=0.1)
* ✅ Чёткое разделение визуальных данных и справочной информации

---

## 🚀 Быстрый старт

### Предварительные требования

* Python 3.10+
* OpenAI API Key (получить на [platform.openai.com](https://platform.openai.com))
* Модель GPT-4.1 с поддержкой vision (например, `gpt-4o` или `gpt-4-turbo`)

### Установка

1. **Клонируйте репозиторий:**

```bash
git clone https://github.com/andreyko75/analize_stamp.git
cd analize_stamp
```

2. **Создайте виртуальное окружение:**

```bash
python3 -m venv venv
```

3. **Активируйте виртуальное окружение:**

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

4. **Установите зависимости:**

```bash
pip install -r requirements.txt
```

5. **Создайте файл `.env` в корне проекта:**

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1
OPENAI_TTS_MODEL=tts-1
OPENAI_TTS_VOICE=alloy
OPENAI_TTS_FORMAT=mp3
```

6. **Запустите анализ:**

```bash
python analyze_stamp.py input/stamp1.jpg
```

Или с генерацией голосового описания:

```bash
python analyze_stamp.py input/stamp1.jpg --tts
```

---

## 🛠 Технологии

* **Python 3.10+** — основной язык
* **OpenAI API** — работа с GPT-4.1 multimodal моделями
* **python-dotenv** — управление переменными окружения

---

## 📁 Структура проекта

```
analize_stamp/
├── input/ # Папка для входных изображений
├── output/ # Папка для результатов (JSON, аудио, тексты)
├── analyze_stamp.py # Основной скрипт анализа
├── json_to_voice.py # Модуль генерации голосового описания
├── requirements.txt # Зависимости проекта
├── .env # Переменные окружения (не в git)
├── .gitignore # Исключения для git
└── README.md # Документация
```

---

_Создано с ❤️ для автоматизации анализа почтовых марок_

**Репозиторий:** [https://github.com/andreyko75/analize_stamp](https://github.com/andreyko75/analize_stamp)
