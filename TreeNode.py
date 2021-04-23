class TreeNode:
    def __init__(self):
        self.isLeaf = False
        self.classLabel = ""
        self.children = []
        self.conditions = []
        self.counts = []

    def set_counts(self, counts):
        self.counts = counts

    def add_child(self, cond, child):
        self.conditions.append(cond)
        self.children.append(child)

    def set_leaf(self, majority_label):
        self.isLeaf = True
        self.classLabel = majority_label

    def print(self, indent=""):
        for cond, child in zip(self.conditions, self.children):
            print_line = indent + str(cond) + " [{} {}]".format(child.counts[0], child.counts[1])
            if child.isLeaf:
                print(print_line + ": " + child.classLabel)
            else:
                print(print_line)
                child.print(indent + "|" + "\t")


class Condition:
    def __init__(self, name, operator, value):
        self.attribute_name = name
        self.operator = operator
        self.value = value

    def __str__(self):
        if type(self.value) != str:
            val = '{:.6f}'.format(self.value)
        else:
            val = self.value
        return "" + self.attribute_name.lower() + " " + self.operator + " " + val

    def is_fit(self, record):
        attribute_value = record[self.attribute_name]
        if self.operator == "=":
            return self.value == attribute_value
        elif self.operator == "<=":
            return attribute_value <= self.value
        else:
            return attribute_value > self.value

