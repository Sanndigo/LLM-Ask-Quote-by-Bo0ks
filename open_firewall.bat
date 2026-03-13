@echo off
echo ============================================================
echo Открытие порта 5000 для Flask RAG
echo ============================================================
echo.

netsh advfirewall firewall add rule name="Flask RAG Port 5000" dir=in action=allow protocol=TCP localport=5000

echo.
echo ============================================================
echo Готово! Порт 5000 открыт
echo ============================================================
pause
