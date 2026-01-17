"""
Postage stamp image analysis using OpenAI GPT-4.1 multimodal capabilities.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from json_to_voice import json_to_voice

# Загружаем переменные окружения из файла .env
load_dotenv()

# Системный промпт для анализа почтовых марок
SYSTEM_PROMPT = """Ты ассистент по анализу изображений почтовых марок и справочному описанию филателистических объектов.

Твоя задача:
1. Проанализировать изображение почтовой марки и извлечь из него основную визуально доступную информацию.
2. Дополнить результат кратким справочным описанием марки на основе общедоступных знаний и открытых источников.
3. Чётко разделять:
   - данные, полученные строго из изображения;
   - справочную информацию, которая может требовать проверки по каталогу.

Правила работы:
1. Используй ТОЛЬКО информацию, которую можно увидеть на изображении, для полей визуального анализа.
2. Справочную информацию формируй ТОЛЬКО как вероятностную и описательную.
3. НЕ выдумывай каталожные номера, тиражи и цены.
4. Если в каких-то данных есть сомнения — явно укажи это.
5. Всегда отмечай, что справочная информация не заменяет данные каталогов.
6. Верни результат СТРОГО в формате JSON, без пояснений и текста вне JSON.

Извлеки визуально доступные данные (если они есть на изображении):

- country — страна или территория выпуска
- postal_type — тип почты (например: обычная, авиапочта и т.п.)
- denomination — номинал и валюта
- year_or_period — год или период выпуска (если можно определить)
- subject — сюжет или тематика изображения
- visible_text — весь читаемый текст на марке одной строкой
- colors — основные доминирующие цвета
- condition_notes — визуальные замечания о состоянии (если применимо)
- uncertainties — список пунктов, где есть сомнения или ограничения анализа
- confidence — общая оценка уверенности распознавания от 0 до 1

Дополнительно сформируй справочный блок reference_info:

- description — краткое текстовое описание марки и события, которому она посвящена
- historical_context — общий исторический или тематический контекст выпуска
- purpose — предполагаемое назначение выпуска (памятный, спортивный, рекламный и т.п.)
- info_source — всегда указывай значение "open sources"
- verification_note — обязательная пометка о необходимости сверки с филателистическими каталогами

Формат ответа:

{
  "country": "...",
  "postal_type": "...",
  "denomination": "...",
  "year_or_period": "...",
  "subject": "...",
  "visible_text": "...",
  "colors": ["...", "..."],
  "condition_notes": "...",
  "uncertainties": ["..."],
  "confidence": 0.0,
  "reference_info": {
    "description": "...",
    "historical_context": "...",
    "purpose": "...",
    "info_source": "open sources",
    "verification_note": "Информация требует сверки с филателистическими каталогами"
  }
}
"""


def encode_image(image_path):
    """
    Кодирует изображение в base64 для передачи в OpenAI API.
    
    Args:
        image_path: Путь к файлу изображения
        
    Returns:
        base64-encoded строка изображения
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_stamp(image_path):
    """
    Анализирует изображение почтовой марки через OpenAI API.
    
    Args:
        image_path: Путь к файлу изображения
        
    Returns:
        JSON-строка с результатами анализа
    """
    # Получаем API ключ и модель из переменных окружения
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден в переменных окружения. Проверьте файл .env")
    
    if not model:
        raise ValueError("OPENAI_MODEL не найден в переменных окружения. Проверьте файл .env")
    
    # Проверяем существование файла изображения
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл изображения не найден: {image_path}")
    
    # Кодируем изображение в base64
    base64_image = encode_image(image_path)
    
    # Определяем MIME-тип изображения по расширению
    image_ext = Path(image_path).suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(image_ext, 'image/jpeg')
    
    # Создаем клиент OpenAI
    client = OpenAI(api_key=api_key)
    
    # Формируем запрос к API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Проанализируй это изображение почтовой марки и верни результат в формате JSON."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.1,  # Детерминированный запрос
        response_format={"type": "json_object"}
    )
    
    # Проверяем наличие ответа в response
    if not response.choices or len(response.choices) == 0:
        raise ValueError("API вернул пустой список choices")
    
    if not response.choices[0].message:
        raise ValueError("API вернул ответ без message")
    
    # Извлекаем ответ модели
    result = response.choices[0].message.content
    
    # Проверяем, что ответ не пустой
    if not result:
        raise ValueError("Модель вернула пустой ответ")
    
    return result


def main():
    """
    Главная функция скрипта.
    Обрабатывает аргументы командной строки, запускает анализ и сохраняет результаты.
    """
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(
        description="Анализ изображения почтовой марки через OpenAI GPT-4.1"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Путь к файлу изображения почтовой марки"
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Генерировать голосовое описание марки после анализа"
    )
    
    args = parser.parse_args()
    
    try:
        # Запускаем анализ
        print(f"Анализирую изображение: {args.image_path}")
        result = analyze_stamp(args.image_path)
        
        # Парсим JSON для валидации
        result_json = json.loads(result)
        
        # Выводим результат в консоль
        print("\nРезультат анализа:")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))
        
        # Сохраняем результат в файл
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "result.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
        
        print(f"\nРезультат сохранен в: {output_file}")
        
        # Генерация голосового описания, если указан флаг --tts
        if args.tts:
            try:
                print("\nГенерирую голосовое описание...")
                script_file, audio_file = json_to_voice(str(output_file), str(output_dir))
                print(f"✓ Текст озвучки сохранен: {script_file}")
                print(f"✓ Аудиофайл сохранен: {audio_file}")
            except Exception as e:
                print(f"Предупреждение: Не удалось сгенерировать голосовое описание: {e}", file=sys.stderr)
        
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка конфигурации: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Ошибка: Не удалось распарсить JSON ответ от модели: {e}", file=sys.stderr)
        print(f"Ответ модели: {result}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
