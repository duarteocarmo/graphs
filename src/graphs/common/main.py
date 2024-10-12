import typing as t
from typing import List

from graphviz import Digraph
from pydantic import BaseModel, Field

from graphs import CLIENT, MODEL


class Node(BaseModel):
    id: int
    name: str
    node_type: t.Literal["PERSON", "PLACE", "ORGANIZATION", "EVENT", "OTHER"]

    def __hash__(self):
        return hash((self.id, self.name, self.node_type))


class Edge(BaseModel):
    source: int
    target: int
    relationship_decription: str

    def __hash__(self):
        return hash((self.source, self.target, self.relationship_decription))


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Updates the current graph with the other graph, deduplicating nodes and edges."""
        return KnowledgeGraph(
            nodes=list(set(self.nodes + other.nodes)),
            edges=list(set(self.edges + other.edges)),
        )

    def draw(self, prefix: str = None):
        dot = Digraph(comment="Knowledge Graph")

        for node in self.nodes:
            dot.node(
                str(node.id), node.name, color="lightblue2", style="filled"
            )

        for edge in self.edges:
            dot.edge(
                str(edge.source),
                str(edge.target),
                label=edge.relationship_decription,
            )
        dot.render(prefix, format="png", view=True)


def generate_graph(input: List[str]) -> KnowledgeGraph:
    cur_state = KnowledgeGraph()

    SYSTEM_PROMPT = """
    You are an iterative knowledge graph builder.
    You are given the current state of the user's graph, and you must append the nodes and edges to it
    Do not produce any duplcates and try to reuse nodes as much as possible.""".strip()

    USER_PROMPT = """
    Extract any new nodes and edges from the following user message:

    # Part {i}/{num_iterations} of the input:

    <user_message>
    {user_message}
    </user_message>
    """.strip()

    num_iterations = len(input)
    for i, inp in enumerate(input):
        completion = CLIENT.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        i=i, num_iterations=num_iterations, user_message=inp
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Here is the current state of the graph:
                    {cur_state.model_dump_json(indent=2)}""",
                },
            ],
            response_format=KnowledgeGraph,
        )

        new_updates = completion.choices[0].message.parsed
        cur_state = cur_state.update(new_updates)

        cur_state.draw(prefix=f"iteration_{i}")
    return cur_state


def visualize_knowledge_graph(kg: KnowledgeGraph):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(
            str(node.id),
            node.name,
            color="lightblue2",
            style="filled",
            shape="box",
        )

    # Add edges
    for edge in kg.edges:
        dot.edge(
            str(edge.source),
            str(edge.target),
            label=edge.relationship_decription,
        )

    # Render the graph
    dot.render("knowledge_graph.gv", view=True)


if __name__ == "__main__":
    text_chunks = [
        "Jason knows a lot about quantum mechanics. He is a physicist. He is a professor",
        "Professors are smart.",
        "Sarah knows Jason and is a student of his.",
        "Sarah is a student at the University of Toronto. and UofT is in Canada",
    ]

    graph: KnowledgeGraph = generate_graph(text_chunks)
    visualize_knowledge_graph(graph)
