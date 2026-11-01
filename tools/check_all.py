"""Run every program in the repository and compare its output with the output printed in the book.

Usage (from the repository root):
    python tools/check_all.py               # check all programs
    python tools/check_all.py --chapter 7   # check one chapter
    python tools/check_all.py --update      # overwrite expected outputs with the current results
"""
import argparse, glob, os, subprocess, sys, difflib
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def find(chapter=None):
    files = sorted(glob.glob(os.path.join(ROOT, 'part*', 'ch*', 'ex*.py')))
    if chapter is not None:
        files = [f for f in files if os.path.basename(os.path.dirname(f)).startswith(f'ch{chapter:02d}_')]
    return files

def run(path):
    folder, name = os.path.split(path)
    expected_file = os.path.join(folder, 'expected_output', name[:-3] + '.txt')
    try:
        r = subprocess.run([sys.executable, name], cwd=folder, capture_output=True, text=True, timeout=900,
                           env={**os.environ, 'PYTHONWARNINGS': 'ignore', 'MPLBACKEND': 'Agg'})
        out, err, code = r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        out, err, code = '', 'timed out after 900 s', -1
    expected = open(expected_file).read() if os.path.exists(expected_file) else None
    same = expected is not None and [l.rstrip() for l in out.strip().splitlines()] == [l.rstrip() for l in expected.strip().splitlines()]
    return path, expected_file, code, same, out, err, expected

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--chapter', type=int); ap.add_argument('--update', action='store_true')
    a = ap.parse_args()
    files = find(a.chapter)
    if not files:
        sys.exit('No programs found.')
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
        results = list(pool.map(run, files))
    failed = 0
    for path, exp_file, code, same, out, err, expected in results:
        rel = os.path.relpath(path, ROOT)
        if code != 0:
            failed += 1; print(f'ERROR    {rel}\n         ' + (err.strip().splitlines() or ['?'])[-1])
        elif a.update:
            open(exp_file, 'w').write(out.rstrip('\n') + '\n'); print(f'UPDATED  {rel}')
        elif same:
            print(f'OK       {rel}')
        else:
            failed += 1; print(f'DIFFERS  {rel}')
            diff = difflib.unified_diff((expected or '').strip().splitlines(), out.strip().splitlines(), 'book', 'now', lineterm='', n=0)
            for line in list(diff)[2:12]: print('         ' + line)
    print(f'\n{len(results) - failed} of {len(results)} programs match the book.')
    sys.exit(1 if failed else 0)

if __name__ == '__main__':
    main()
