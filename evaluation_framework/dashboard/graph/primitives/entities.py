
class Node:
    def __init__(self, _id: int, _label: str, _classes: str = None, **kwargs):
        self.id = _id
        self.label = _label
        self.classes = _classes
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.data = {k: v for k, v in self.__dict__.items() if k not in ['classes']}


class Link:
    def __init__(self, source: Node, target: Node, weight: float = 1.0, classes: str = None):
        self.source = source
        self.target = target
        self.weight = weight
        self.classes = classes
        self.data = {
            "source": self.source.id,
            "target": self.target.id,
            "weight": self.weight,
        }
