import subprocess
import time


def kill_port(port):
    try:
        output = subprocess.check_output(
            f"netstat -ano | findstr :{port}", shell=True
        ).decode()
        for line in output.splitlines():
            if "LISTENING" in line:
                pid = line.strip().split()[-1]
                print(f"Killing process {pid} on port {port}")
                subprocess.run(f"taskkill /PID {pid} /F", shell=True)
    except Exception:
        pass


print("Cleaning up ports 8000 and 3000...")
kill_port(8000)
kill_port(3000)
kill_port(3001)

time.sleep(2)

print("Starting Backend...")
# Use start to run in separate window or just rely on Gemini background
# Since I'm in a python script, I'll just exit and let Gemini start them cleanly.
