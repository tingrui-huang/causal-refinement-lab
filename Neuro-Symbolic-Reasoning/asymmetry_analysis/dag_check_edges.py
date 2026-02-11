import sys
from collections import defaultdict, deque
from pathlib import Path

def load_edges_from_complete_edges_txt(p: Path):
    edges = []
    nodes = set()
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("Learned") or line.startswith("="):
            continue
        if "->" not in line:
            continue
        left, right = line.split("->", 1)
        u = left.strip()
        v = right.split("(strength:", 1)[0].strip()
        if not u or not v:
            continue
        edges.append((u, v))
        nodes.add(u); nodes.add(v)
    return nodes, edges

def is_dag_kahn(nodes, edges):
    g = defaultdict(list)
    indeg = {n: 0 for n in nodes}
    for u, v in edges:
        g[u].append(v)
        indeg[v] = indeg.get(v, 0) + 1
        indeg.setdefault(u, 0)

    q = deque([n for n, d in indeg.items() if d == 0])
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in g.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen == len(indeg)

def find_one_cycle(nodes, edges):
    g = defaultdict(list)
    for u, v in edges:
        g[u].append(v)

    color = {n: 0 for n in nodes}  # 0=unseen,1=visiting,2=done
    parent = {}

    def dfs(u):
        color[u] = 1
        for v in g.get(u, []):
            if color.get(v, 0) == 0:
                parent[v] = u
                cyc = dfs(v)
                if cyc: return cyc
            elif color.get(v) == 1:
                # back edge u->v, reconstruct cycle v ... u -> v
                path = [v]
                cur = u
                while cur != v and cur in parent:
                    path.append(cur)
                    cur = parent[cur]
                path.append(v)
                path.reverse()
                return path
        color[u] = 2
        return None

    for n in nodes:
        if color.get(n, 0) == 0:
            parent[n] = None
            cyc = dfs(n)
            if cyc: return cyc
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python dag_check_edges.py <complete_edges.txt> [more...]")
        sys.exit(2)

    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            p = p / "complete_edges.txt"
        if not p.exists():
            print(f"[ERROR] Not found: {p}")
            continue

        nodes, edges = load_edges_from_complete_edges_txt(p)
        dag = is_dag_kahn(nodes, edges)
        print(f"\n=== DAG CHECK ===")
        print(f"path: {p}")
        print(f"nodes: {len(nodes)}, edges: {len(edges)}")
        print(f"is_dag: {dag}")
        if not dag:
            cyc = find_one_cycle(nodes, edges)
            print("cycle_example:", " -> ".join(cyc) if cyc else "(not found)")
        print("===============")

if __name__ == "__main__":
    main()