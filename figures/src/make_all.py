"""Redraw all 99 figures of the book into figures/png/.  Usage: python figures/src/make_all.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
for m in ('part1', 'part2', 'part3', 'part4'):
    __import__(m)
print(f'Drew {len(common.REG)} figures into', os.path.abspath(common.OUT))
