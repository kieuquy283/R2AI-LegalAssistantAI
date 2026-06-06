from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from src.ingestion.common import read_jsonl


class LegalGraphStore:
    def __init__(
        self,
        *,
        nodes_path: str | Path = "data/processed/legal_graph_nodes.jsonl",
        edges_path: str | Path = "data/processed/legal_graph_edges.jsonl",
    ) -> None:
        self.nodes = {
            str(row["node_id"]): row
            for row in read_jsonl(nodes_path)
        }
        self.edges = read_jsonl(edges_path)
        self.outgoing: Dict[str, List[Dict[str, object]]] = {}
        self.incoming: Dict[str, List[Dict[str, object]]] = {}
        for edge in self.edges:
            source_id = str(edge.get("source_id") or "")
            target_id = str(edge.get("target_id") or "")
            self.outgoing.setdefault(source_id, []).append(edge)
            self.incoming.setdefault(target_id, []).append(edge)

    def get_node(self, node_id: str) -> Dict[str, object] | None:
        return self.nodes.get(str(node_id))

    def _iter_edges(
        self,
        node_id: str,
        *,
        relation_types: Sequence[str] | None = None,
        include_reverse: bool = True,
    ) -> Iterable[tuple[Dict[str, object], str]]:
        allowed = set(relation_types or [])
        for edge in self.outgoing.get(str(node_id), []):
            if allowed and str(edge.get("relation_type")) not in allowed:
                continue
            yield edge, str(edge.get("target_id") or "")
        if include_reverse:
            for edge in self.incoming.get(str(node_id), []):
                if allowed and str(edge.get("relation_type")) not in allowed:
                    continue
                yield edge, str(edge.get("source_id") or "")

    def get_neighbors(
        self,
        node_id: str,
        relation_types: Sequence[str] | None = None,
        max_depth: int = 1,
    ) -> List[Dict[str, object]]:
        frontier = {str(node_id)}
        visited = {str(node_id)}
        neighbors: List[Dict[str, object]] = []
        seen_neighbor_ids = set()

        for _depth in range(max(1, int(max_depth))):
            next_frontier = set()
            for current_id in frontier:
                for edge, neighbor_id in self._iter_edges(current_id, relation_types=relation_types):
                    if not neighbor_id or neighbor_id == str(node_id):
                        continue
                    if neighbor_id not in seen_neighbor_ids:
                        neighbors.append(
                            {
                                "node_id": neighbor_id,
                                "node": self.get_node(neighbor_id),
                                "via_edge": edge,
                            }
                        )
                        seen_neighbor_ids.add(neighbor_id)
                    if neighbor_id not in visited:
                        next_frontier.add(neighbor_id)
                        visited.add(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break
        return neighbors

    def get_parent(self, node_id: str) -> Dict[str, object] | None:
        for edge in self.outgoing.get(str(node_id), []):
            if str(edge.get("relation_type")) == "HAS_PARENT":
                return self.get_node(str(edge.get("target_id") or ""))
        return None

    def get_children(self, node_id: str) -> List[Dict[str, object]]:
        children: List[Dict[str, object]] = []
        seen = set()
        for edge in self.outgoing.get(str(node_id), []):
            if str(edge.get("relation_type")) != "HAS_CHILD":
                continue
            child = self.get_node(str(edge.get("target_id") or ""))
            child_id = str(edge.get("target_id") or "")
            if child_id and child_id not in seen:
                children.append(child or {"node_id": child_id})
                seen.add(child_id)
        for edge in self.incoming.get(str(node_id), []):
            if str(edge.get("relation_type")) != "HAS_PARENT":
                continue
            child = self.get_node(str(edge.get("source_id") or ""))
            child_id = str(edge.get("source_id") or "")
            if child_id and child_id not in seen:
                children.append(child or {"node_id": child_id})
                seen.add(child_id)
        return children

    def get_explicit_refs(self, node_id: str) -> List[Dict[str, object]]:
        return [
            edge
            for edge in self.outgoing.get(str(node_id), [])
            if str(edge.get("relation_type")) == "REFERS_TO"
        ]

    def get_cross_domains(self, node_id: str) -> List[Dict[str, object]]:
        return [
            edge
            for edge in self.outgoing.get(str(node_id), [])
            if str(edge.get("relation_type")) == "CROSS_DOMAIN"
        ]


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect canonical legal graph store.")
    parser.add_argument("--nodes", default="data/processed/legal_graph_nodes.jsonl")
    parser.add_argument("--edges", default="data/processed/legal_graph_edges.jsonl")
    parser.add_argument("--node-id", default=None)
    args = parser.parse_args()
    store = LegalGraphStore(nodes_path=args.nodes, edges_path=args.edges)
    if args.node_id:
        payload = {
            "node": store.get_node(args.node_id),
            "neighbors": store.get_neighbors(args.node_id),
        }
    else:
        payload = {"nodes": len(store.nodes), "edges": len(store.edges)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
