import ast
for f in ['backend/main.py', 'src/ood/ood_guards.py']:
    try:
        ast.parse(open(f).read())
        print(f"{f}: syntax OK")
    except SyntaxError as e:
        print(f"{f}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
