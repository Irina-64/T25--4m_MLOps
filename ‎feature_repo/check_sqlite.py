# feature_repo/check_sqlite.py
import sqlite3
import pandas as pd

print("=== ПРОВЕРКА SQLITE БАЗЫ ДАННЫХ ===")

try:
    # Подключаемся к базе
    conn = sqlite3.connect('data/online_store.db')
    cursor = conn.cursor()
    
    # 1. Какие таблицы есть?
    print("\n1. 📋 ТАБЛИЦЫ В БАЗЕ:")
    cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
    tables = cursor.fetchall()
    
    if not tables:
        print("   ❌ Нет таблиц! База данных пуста.")
    else:
        for table in tables:
            print(f"   - {table[0]}")
    
    # 2. Проверяем таблицу currency_features
    print("\n2. 🔍 ПРОВЕРКА ТАБЛИЦЫ CURRENCY_FEATURES:")
    
    # Ищем таблицу с currency
    currency_tables = [t[0] for t in tables if 'currency' in t[0].lower()]
    
    if not currency_tables:
        print("   ❌ Таблица currency_features не найдена!")
    else:
        for table_name in currency_tables:
            print(f"\n   Таблица: {table_name}")
            
            # Количество строк
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count = cursor.fetchone()[0]
            print(f"   Всего строк: {count}")
            
            if count > 0:
                # Столбцы таблицы
                cursor.execute(f'PRAGMA table_info("{table_name}")')
                columns = cursor.fetchall()
                print(f"   Столбцы: {[col[1] for col in columns]}")
                
                # Первые 3 строки
                cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
                rows = cursor.fetchall()
                print(f"   Первые {len(rows)} строк:")
                for row in rows:
                    print(f"     {row}")
            else:
                print("   ⚠️ Таблица пустая!")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Ошибка: {e}")