class Node:
    def __init__(self, label=None, feature=None, children=None, modal_class=None):
        self.label = label
        self.feature = feature # nazwa atrybutu, na podstawie którego dokonano podziału
        self.children = children if children is not None else {}
        self.modal_class = modal_class
        # {
        #     1.0: Node_Potomek_1,
        #     2.0: Node_Potomek_2,
        #     3.0: Node_Potomek_3
        # }

    def is_leaf(self):
        return len(self.children) == 0

    def get_children(self):
        return self.children

    def get_label(self):
        return self.label

    def __repr__(self):
        if self.is_leaf():
            return f"Liść ({self.label})"
        else:
            return f"Węzeł({self.feature}, podział: {list(self.children.keys())})"