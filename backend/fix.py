import glob

# Fix except Exception:
files = glob.glob('**/*.py', recursive=True)
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'except Exception:' in content:
        content = content.replace('except Exception:', 'except Exception:')
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)

# Fix backtester.py pd import
with open('src/execution/backtester.py', 'r', encoding='utf-8') as f:
    content = f.read()
if 'import pandas as pd' not in content:
    content = 'import pandas as pd\n' + content
    with open('src/execution/backtester.py', 'w', encoding='utf-8') as f:
        f.write(content)

# Fix empirical_validation.py import json
with open('src/execution/empirical_validation.py', 'r', encoding='utf-8') as f:
    content = f.read()
if 'import json' not in content:
    content = 'import json\n' + content
    with open('src/execution/empirical_validation.py', 'w', encoding='utf-8') as f:
        f.write(content)

# Fix backtest_service.py E712
with open('src/execution/backtest_service.py', 'r', encoding='utf-8') as f:
    content = f.read()
if "df_ticker['was_correct'] == True" in content:
    content = content.replace("df_ticker['was_correct'] == True", "df_ticker['was_correct']")
    with open('src/execution/backtest_service.py', 'w', encoding='utf-8') as f:
        f.write(content)
