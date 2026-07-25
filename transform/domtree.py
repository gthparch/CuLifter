from collections import defaultdict


class DominatorTree:
    def __init__(self, entry_block, blocks):
        self.entry = entry_block
        self.blocks = blocks
        self.idom = {}
        self.children = defaultdict(list)
        self.dominance_frontier = defaultdict(set)
        self._dfs_order = []
        self._dfs_index = {}

        self.compute_dominators()
        self.build_tree()
        self.compute_dominance_frontiers()

    def compute_dominators(self):
        visited = set()
        order = []

        stack = []
        if self.entry:
            stack.append((self.entry, iter(self.entry.succs)))
            visited.add(self.entry)

        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(child.succs)))
            except StopIteration:
                stack.pop()
                order.append(parent)

        rpo = list(reversed(order))
        for idx, b in enumerate(rpo):
            self._dfs_index[b] = idx

        self.idom = {self.entry: self.entry}
        changed = True

        def intersect(b1, b2):
            finger1, finger2 = b1, b2
            while finger1 != finger2:
                while self._dfs_index[finger1] > self._dfs_index[finger2]:
                    finger1 = self.idom[finger1]
                while self._dfs_index[finger2] > self._dfs_index[finger1]:
                    finger2 = self.idom[finger2]
            return finger1

        while changed:
            changed = False
            for b in rpo:
                if b is self.entry:
                    continue
                preds = [p for p in b.preds if p in self._dfs_index and p in self.idom]
                if not preds:
                    continue
                new_idom = None
                for p in preds:
                    if p in self.idom:
                        new_idom = p if new_idom is None else intersect(p, new_idom)
                if b not in self.idom or self.idom[b] is not new_idom:
                    self.idom[b] = new_idom
                    changed = True

    def build_tree(self):
        for block in list(self.children.keys()):
            self.children[block].clear()
        for block, idom_block in self.idom.items():
            if block is idom_block:
                continue
            self.children[idom_block].append(block)

    def compute_dominance_frontiers(self):
        self.dominance_frontier = defaultdict(set)
        for b in self.blocks:
            if len(b.preds) < 2:
                continue
            for p in b.preds:
                runner = p
                while runner is not None and runner != self.idom.get(b):
                    self.dominance_frontier[runner].add(b)
                    runner = self.idom.get(runner)

    def preorder(self):
        result = []
        stack = [self.entry]

        while stack:
            block = stack.pop()
            result.append(block)

            for child in reversed(self.children[block]):
                stack.append(child)

        return result

    def get_children(self, block):
        return self.children[block]

