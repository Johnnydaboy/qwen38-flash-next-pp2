import urllib.request
import json
import time

prompt = "hello world " * 200 + "\nWrite an exceptionally detailed science fiction novel chapter about a deep-space expedition exploring the outer boundaries of a shattered Dyson sphere."

payload = {
    "model": "qwen38-flash-next-awq",
    "prompt": prompt,
    "max_tokens": 3000,
    "temperature": 0.0,
    "stream": False
}

print(f"Sending request with max_tokens={payload['max_tokens']}...")
t0 = time.time()
req = urllib.request.Request(
    "http://192.168.1.30:8001/v1/completions",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload).encode("utf-8")
)

try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        text = result["choices"][0]["text"]
        usage = result["usage"]
        duration = time.time() - t0
        tok_per_sec = usage["completion_tokens"] / duration
        print(f"Completed in {duration:.2f}s ({tok_per_sec:.1f} tok/s)")
        print(f"Usage: {usage}")
        print(f"Output length: {len(text)} chars")
        print("\nLast 300 chars:")
        print(repr(text[-300:]))
        
        bang_count = 0
        for ch in reversed(text):
            if ch == '!':
                bang_count += 1
            else:
                break
        print(f"\nTrailing exclamation marks count: {bang_count}")
        
        # Check repetition of phrases
        words = text.split()
        if len(words) > 50:
            last_50 = " ".join(words[-50:])
            print(f"\nLast 50 words snippet:\n{last_50}")
except Exception as e:
    print(f"Error: {e}")
