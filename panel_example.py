import typing as t

import panel as pn
import param
from graphviz import Digraph
from litellm import acompletion
from panel.viewable import Viewer
from pydantic import BaseModel, Field

from graphs import MODEL

pn.extension(sizing_mode="stretch_width")

CSS = """
div.card-margin:nth-child(1) {
    max-height: 30vh;
}
div.card-margin:nth-child(2) {
    max-height: 55vh;
}
"""


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
    nodes: list[Node] = Field(..., default_factory=list)
    edges: list[Edge] = Field(..., default_factory=list)

    def update(self, other: "KnowledgeGraph") -> "KnowledgeGraph":
        """Updates the current graph with the other graph, deduplicating nodes and edges."""
        return KnowledgeGraph(
            nodes=list(set(self.nodes + other.nodes)),
            edges=list(set(self.edges + other.edges)),
        )

    def draw(self, prefix: str = None):
        dot = Digraph(comment="Knowledge Graph", format="svg")
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
        svg_content = dot.pipe(encoding="utf-8")
        svg_content = svg_content[svg_content.find("<svg") :]
        return pn.pane.SVG(svg_content, sizing_mode="stretch_both")


class KnowledgeGraphApp(Viewer):
    graph = param.ClassSelector(
        class_=KnowledgeGraph, default=KnowledgeGraph()
    )

    def __init__(self, **params):
        super().__init__(**params)

        self.graph = self.create_example_graph("Hello World")
        self.chat_interface = pn.chat.ChatInterface(
            callback=self.chat_callback,
            callback_user="Bot",
            callback_avatar="🤖",
            help_text="Send a message",
            sizing_mode="stretch_both",
        )

    def create_example_graph(self, text: str):
        graph = KnowledgeGraph()
        # Add nodes
        graph.nodes.extend(
            [
                Node(id=1, name=text, node_type="PERSON"),
                Node(id=2, name="New York City", node_type="PLACE"),
                Node(id=3, name="Acme Corp", node_type="ORGANIZATION"),
                Node(id=4, name="2024 Tech Conference", node_type="EVENT"),
            ]
        )
        # Add edges
        graph.edges.extend(
            [
                Edge(source=1, target=2, relationship_decription="lives in"),
                Edge(source=1, target=3, relationship_decription="works at"),
                Edge(source=1, target=4, relationship_decription="attended"),
                Edge(source=3, target=4, relationship_decription="sponsored"),
            ]
        )
        return graph

    async def chat_callback(
        self, contents: str, user: str, instance: pn.chat.ChatInterface
    ):
        messages = instance.serialize()
        response = await acompletion(
            model=MODEL,
            messages=messages,
            stream=True,
            # tools=[schema(self.update_knowledge_base)],
            # tool_choice="auto",
        )
        message = ""
        async for chunk in response:
            part = chunk.choices[0].delta.content
            if part is not None:
                message += part
                yield message

        self.graph = self.create_example_graph(contents)

    @param.depends("graph", watch=False)
    def _update_svg(self):
        return self.graph.draw()

    def __panel__(self):
        return pn.template.FastListTemplate(
            title="Knowledge Graph App",
            main=[self._update_svg, self.chat_interface],
            header=[],
            theme_toggle=True,
            busy_indicator=None,
            raw_css=[CSS],
        )


app = KnowledgeGraphApp()
app.servable()
