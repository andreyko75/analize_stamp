"""
Модуль для генерации голосового описания почтовой марки на основе JSON-результата анализа.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения из файла .env
load_dotenv()

# Промпт для генерации текста озвучки
VOICE_SCRIPT_PROMPT = """Ты создаёшь текст для голосового описания почтовой марки на русском языке.

Требования к тексту:
1. Дружелюбный, спокойный стиль, как голосовой помощник филателиста
2. Без канцелярита, без рекламы
3. 4-7 коротких фраз, общая длительность 20-40 секунд
4. Связный, живой текст, НЕ сухой перечень полей
5. НЕ выдумывай каталожные номера, тиражи, цены, редкость
6. Если есть uncertainties - обязательно упомяни их в конце одной фразой
7. Если есть reference_info - используй её, но обязательно проговори, что справочная часть из открытых источников и требует сверки с каталогом (кратко, одним предложением)

На основе следующих данных о марке создай текст для озвучки:

{stamp_data}

Верни ТОЛЬКО текст для озвучки, без пояснений и метаданных."""


def load_json_result(json_path):
    """
    Загружает JSON-результат анализа марки из файла.
    
    Args:
        json_path: Путь к JSON-файлу
        
    Returns:
        dict: Данные о марке
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON-файл не найден: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_voice_script(stamp_data, api_key, model):
    """
    Генерирует текст для озвучки на основе данных о марке.
    
    Args:
        stamp_data: dict с данными о марке
        api_key: OpenAI API ключ
        model: Модель OpenAI для генерации текста
        
    Returns:
        str: Текст для озвучки
    """
    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден в переменных окружения. Проверьте файл .env")
    
    if not model:
        raise ValueError("OPENAI_MODEL не найден в переменных окружения. Проверьте файл .env")
    
    # Формируем строку с данными о марке
    data_str = json.dumps(stamp_data, ensure_ascii=False, indent=2)
    
    # Создаем клиент OpenAI
    client = OpenAI(api_key=api_key)
    
    # Генерируем текст для озвучки
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": VOICE_SCRIPT_PROMPT
            },
            {
                "role": "user",
                "content": f"Данные о марке:\n{data_str}"
            }
        ],
        temperature=0.2
    )
    
    # Извлекаем текст
    if not response.choices or len(response.choices) == 0:
        raise ValueError("API вернул пустой список choices")
    
    if not response.choices[0].message:
        raise ValueError("API вернул ответ без message")
    
    voice_script = response.choices[0].message.content
    
    if not voice_script:
        raise ValueError("Модель вернула пустой текст для озвучки")
    
    return voice_script.strip()


def generate_audio(voice_script, api_key, tts_model, tts_voice, tts_format="mp3"):
    """
    Генерирует аудиофайл из текста через OpenAI TTS.
    
    Args:
        voice_script: Текст для озвучки
        api_key: OpenAI API ключ
        tts_model: Модель TTS
        tts_voice: Голос для озвучки
        tts_format: Формат аудио (mp3, opus, aac, flac)
        
    Returns:
        bytes: Аудио данные
    """
    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден в переменных окружения. Проверьте файл .env")
    
    if not tts_model:
        raise ValueError("OPENAI_TTS_MODEL не найден в переменных окружения. Проверьте файл .env")
    
    if not tts_voice:
        raise ValueError("OPENAI_TTS_VOICE не найден в переменных окружения. Проверьте файл .env")
    
    # Создаем клиент OpenAI
    client = OpenAI(api_key=api_key)
    
    # Генерируем аудио
    response = client.audio.speech.create(
        model=tts_model,
        voice=tts_voice,
        input=voice_script,
        response_format=tts_format
    )
    
    # Получаем аудио данные
    audio_data = response.content
    
    if not audio_data:
        raise ValueError("Не удалось сгенерировать аудио")
    
    return audio_data


def json_to_voice(json_path, output_dir="output"):
    """
    Основная функция: читает JSON, генерирует текст и аудио, сохраняет файлы.
    
    Args:
        json_path: Путь к JSON-файлу с результатами анализа
        output_dir: Директория для сохранения результатов
        
    Returns:
        tuple: (путь к voice_script.txt, путь к result.mp3)
    """
    # Получаем переменные окружения
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    tts_model = os.getenv("OPENAI_TTS_MODEL", "tts-1")
    tts_voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
    tts_format = os.getenv("OPENAI_TTS_FORMAT", "mp3")
    
    # Проверяем обязательные переменные
    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден в переменных окружения. Проверьте файл .env")
    
    if not model:
        raise ValueError("OPENAI_MODEL не найден в переменных окружения. Проверьте файл .env")
    
    # Загружаем JSON
    stamp_data = load_json_result(json_path)
    
    # Генерируем текст для озвучки
    voice_script = generate_voice_script(stamp_data, api_key, model)
    
    # Генерируем аудио
    audio_data = generate_audio(voice_script, api_key, tts_model, tts_voice, tts_format)
    
    # Создаем директорию output, если её нет
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Сохраняем текст озвучки
    script_file = output_path / "voice_script.txt"
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(voice_script)
    
    # Сохраняем аудио
    audio_file = output_path / f"result.{tts_format}"
    with open(audio_file, "wb") as f:
        f.write(audio_data)
    
    return str(script_file), str(audio_file)


def main():
    """
    CLI-интерфейс для запуска модуля отдельно.
    """
    parser = argparse.ArgumentParser(
        description="Генерация голосового описания марки из JSON-результата"
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Путь к JSON-файлу с результатами анализа марки"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Директория для сохранения результатов (по умолчанию: output)"
    )
    
    args = parser.parse_args()
    
    try:
        script_file, audio_file = json_to_voice(args.json_path, args.output_dir)
        print(f"✓ Текст озвучки сохранен: {script_file}")
        print(f"✓ Аудиофайл сохранен: {audio_file}")
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка конфигурации: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
