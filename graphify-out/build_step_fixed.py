
import json
import networkx as nx
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json, to_html
from pathlib import Path

out_dir = Path(r'D:\Projects\Data_Science_Projects\Stock_Indicator\graphify-out')
base_dir = r'D:\Projects\Data_Science_Projects\Stock_Indicator'

extraction = json.loads((out_dir / ".graphify_extract.json").read_text())
detection  = json.loads((out_dir / ".graphify_detect.json").read_text())

G = build_from_json(extraction)
communities = cluster(G)
cohesion = score_all(G, communities)
tokens = {'input': 0, 'output': 0}
gods = god_nodes(G)
surprises = surprising_connections(G, communities)

# Simple labeling
labels = {cid: f"Community {cid}" for cid in communities}

# Attempt to find better labels if possible (simplified for this script)
# In a real graphify run, this would use LLM, but here we'll use most central node labels
for cid, nodes in communities.items():
    subG = G.subgraph(nodes)
    if len(subG) > 0:
        centrality = nx.degree_centrality(subG)
        most_central = max(centrality, key=centrality.get)
        label = G.nodes[most_central].get('label', G.nodes[most_central].get('id', str(cid)))
        labels[cid] = f"{label} Cluster"

questions = suggest_questions(G, communities, labels)

report = generate(G, communities, cohesion, labels, gods, surprises, detection, tokens, base_dir, suggested_questions=questions)
(out_dir / 'GRAPH_REPORT.md').write_text(report)
to_json(G, communities, str(out_dir / 'graph.json'))
to_html(G, communities, str(out_dir / 'graph.html'), community_labels=labels)

analysis = {
    'communities': {str(k): v for k, v in communities.items()},
    'cohesion': {str(k): v for k, v in cohesion.items()},
    'gods': gods,
    'surprises': surprises,
    'questions': questions,
}
(out_dir / '.graphify_analysis.json').write_text(json.dumps(analysis, indent=2))
print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities')
