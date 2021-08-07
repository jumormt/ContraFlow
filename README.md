# ICSE22

```py
PYTHONPATH="." python src/vul_detect.py -c configs/vul_detect.yaml
```

value flow sample 
[st len, str len, value flow len], value flow len

value flow batch 
[st len, str len, total value flow len] [value flow len1, ...]
[total value flow len, hidden dim]
[2N, value flow max len, hidden dim]
[2N, hidden dim]

method sample [st len, str len, total value flow len] [value flow len1, ...] method len=sizeof([value flow len1, ...])

method batch [st len, str len, total value flaw len] [[value flow len1, ...],[value flow len1, ...]] [method len1, ...]
[total value flaw len, hidden dim]
[total method len, value flow max len, hidden dim]
[total method len, hidden dim]
[N, method max len, hidden dim]
[N, hidden dim]


value flow sample 
[AST1, ...]

value flow batch
[AST1, ...] [value flow len1, ...]