import urllib.request
import json
import time

prompt = "hello world " * 200 + "\nWrite a long story about a space voyager exploring distant galaxies."

payload = {
    "model": "qwen38-flash-next-awq",
    "prompt": prompt,
    "max_tokens": 1200,
    "temperature": 0.0,
    "stream": False
}

print(f"Sending request with {len(prompt.split())} words...")
t0 = time.time()
req = urllib.request.Request(
    "http://192.168.1.30:8001/v1/completions",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload).encode("utf-8")
)

try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        text = result["choices"][0]["text"]
        usage = result["usage"]
        print(f"Completed in {time.time()-t0:.2f}s")
        print(f"Usage: {usage}")
        print(f"Output length: {len(text)} chars")
        print("Last 200 chars:")
        print(repr(text[-200:]))
        bang_count = 0
        for ch in reversed(text):
            if ch == '!':
                bang_count += 1
            else:
                break
        print(f"Trailing exclamation marks count: {bang_count}")
except Exception as e:
    print(f"Error: {e}")
