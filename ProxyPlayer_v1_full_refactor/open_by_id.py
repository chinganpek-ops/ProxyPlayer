#!/usr/bin/env python3
"""
Скрипт для открытия плеера по ID через HTTP (ProxyPlayer v1).
Запускает менеджер, отправляет запрос с ID на порт 18080.
Пример: python open_by_id.py 1885054
"""

import sys
import os
import time
import subprocess
import http.client
import threading

MANAGER_CMD = [sys.executable, "main.py"]
HOST = "localhost"
PORT = 18080
WAIT_PORT_TIMEOUT = 10
HTTP_TIMEOUT = 5


def wait_for_port(host, port, timeout):
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def send_http_request(file_id):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=HTTP_TIMEOUT)
    start = time.time()
    try:
        conn.request("GET", f"/open?file={file_id}")
        resp = conn.getresponse()
        body = resp.read().decode()
        elapsed = time.time() - start
        return resp.status, body, elapsed
    except Exception as e:
        elapsed = time.time() - start
        return None, str(e), elapsed
    finally:
        conn.close()


def read_process_output(proc, prefix):
    for line in proc.stdout:
        print(f"[{prefix}] {line}", end='')
    proc.stdout.close()


def main():
    if len(sys.argv) < 2:
        print("Использование: python open_by_id.py <ID_файла>")
        sys.exit(1)
    file_id = sys.argv[1]

    print(f"Запуск менеджера: {' '.join(MANAGER_CMD)}")
    proc = subprocess.Popen(
        MANAGER_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )

    t = threading.Thread(target=read_process_output, args=(proc, "MANAGER"), daemon=True)
    t.start()

    print(f"Ожидание открытия порта {PORT}...")
    if not wait_for_port(HOST, PORT, WAIT_PORT_TIMEOUT):
        print(f"Порт {PORT} не открылся. Менеджер не запустился.")
        proc.terminate()
        sys.exit(1)
    print(f"Порт {PORT} открыт, менеджер готов.")

    time.sleep(1)

    print(f"Отправка запроса для ID={file_id}...")
    status, body, elapsed = send_http_request(file_id)
    if status is not None:
        print(f"HTTP ответ: {status} {body} (время ответа: {elapsed:.2f}с)")
    else:
        print(f"Ошибка HTTP запроса: {body} (время: {elapsed:.2f}с)")

    print("Менеджер продолжает работать. Для выхода нажмите Ctrl+C.")
    try:
        while proc.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nЗавершение...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Менеджер остановлен.")


if __name__ == "__main__":
    main()