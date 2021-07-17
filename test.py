from dataclasses import dataclass

@dataclass
class TT:
    name: str=None

a = TT()

print(a.name)