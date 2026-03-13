from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv('GIGACHAT_CLIENT_ID')
client_secret = os.getenv('GIGACHAT_CLIENT_SECRET')

print("=" * 60)
print("ПРОВЕРКА .env ФАЙЛА")
print("=" * 60)
print(f"\nClient ID: {client_id[:8] if client_id else 'NOT FOUND'}...")
print(f"Client Secret: {client_secret[:8] if client_secret else 'NOT FOUND'}...")

if not client_id or not client_secret:
    print("\n❌ Ключи не найдены в .env!")
    exit(1)

print("\n✅ Ключи найдены")
print("\n" + "=" * 60)
print("ТЕСТ ПОДКЛЮЧЕНИЯ")
print("=" * 60)

import requests

auth_url = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
    'RqUID': '00000000-0000-0000-0000-000000000000',
    'Authorization': f'Bearer {client_secret}'
}
data = {'scope': 'GIGACHAT_API_PERS'}

try:
    response = requests.post(auth_url, headers=headers, data=data, verify=False, timeout=10)
    print(f"\nStatus: {response.status_code}")
    print(f"Response: {response.text[:200]}")
    
    if response.status_code == 200:
        token = response.json()['access_token']
        print(f"\n✅ ТОКЕН ПОЛУЧЕН!")
        print(f"   {token[:50]}...")
    else:
        print(f"\n❌ Ошибка: {response.status_code}")
        print(f"   {response.text}")
except Exception as e:
    print(f"\n❌ Ошибка: {e}")

print("\n" + "=" * 60)
