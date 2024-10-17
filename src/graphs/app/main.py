import inspect
import json
import typing as t
from inspect import Parameter

import litellm
import panel as pn
import param
from graphviz import Digraph
from litellm import acompletion, completion, stream_chunk_builder
from panel.chat import ChatMessage
from panel.viewable import Viewer
from pydantic import BaseModel, Field, create_model

from graphs import MODEL

litellm.enable_json_schema_validation = True
pn.extension(sizing_mode="stretch_width")


def schema(f) -> dict:
    kw = {
        n: (o.annotation, ... if o.default == Parameter.empty else o.default)
        for n, o in inspect.signature(f).parameters.items()
    }
    s = create_model(f"Input for `{f.__name__}`", **kw).schema()

    return {
        "type": "function",
        "function": {
            "name": f.__name__,
            "description": f.__doc__,
            "parameters": s,
        },
    }


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=Noto+Sans:ital,wght@0,100..900;1,100..900&display=swap');

div.card-margin:nth-child(1) {
    max-height: 30vh;
}

div.card-margin:nth-child(2) {
    max-height: 45vh;
}

.avatar {
    display: none !important;
}

:root {
    --body-font: "Noto Sans", sans-serif !important;
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


def update_graph(input: list[str], kg: KnowledgeGraph) -> KnowledgeGraph:
    SYSTEM_PROMPT = """
    You are an iterative knowledge graph builder.
    You are given the current state of the user's graph, and you must append the nodes and edges to it
    Do not produce any duplcates and try to reuse nodes as much as possible.""".strip()

    USER_PROMPT = """
    Extract any new nodes and edges from the following user message:


    <user_message>
    {user_message}
    </user_message>
    """.strip()

    response = completion(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(user_message=input),
            },
            {
                "role": "user",
                "content": f"""Here is the current state of the graph:
                {kg.model_dump_json(indent=2)}""",
            },
        ],
        response_format=KnowledgeGraph,
    )

    try:
        return KnowledgeGraph.model_validate_json(
            response.choices[0].message.content
        )
    except Exception as e:
        raise ValueError(f"Error updating graph: {e}")


class KnowledgeGraphApp(Viewer):
    graph = param.ClassSelector(
        class_=KnowledgeGraph, default=KnowledgeGraph()
    )

    def __init__(self, **params):
        super().__init__(**params)

        self.graph = KnowledgeGraph(
            nodes=[Node(id=0, name="User", node_type="PERSON")]
        )

        self.chat_interface = pn.chat.ChatInterface(
            callback=self.chat_callback,
            callback_user="Bot",
            show_rerun=False,
            help_text="Start chatting and see the knowledge graph update..",
            sizing_mode="stretch_both",
            avatar="",
            default_avatars={
                "System": "S",
                "User": "👤",
                "Bot": "🍕",
                "Help": "ℹ️",
            },
            message_params={
                "show_timestamp": False,
                "show_avatar": False,
                "show_user": False,
                "reaction_icons": {},
                "show_copy_icon": False,
                "stylesheets": [
                    """
                .message {
                    font-size: 0.9em;
                }
                """
                ],
            },
            callback_exception="verbose",
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

    @staticmethod
    def update_knowledge_base(information_about_user: str) -> str:
        """
        Update the knowledge base with information about the user.
        Returns a message indicating that the knowledge base has been updated.
        """
        print(f"Updating knowledge base with: {information_about_user}")
        return information_about_user

    @staticmethod
    def custom_serializer(
        instance: pn.chat.ChatInterface,
    ) -> list[dict[str, str]]:
        messages = []
        for message in instance:
            if message.user == "User":
                messages.append(
                    {
                        "role": "user",
                        "content": message.object,
                    }
                )
            elif message.user == "Bot":
                if isinstance(message.object, str):
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.object,
                        }
                    )
                elif isinstance(message.object, dict):
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": message.object["tool_calls"],
                        }
                    )
                else:
                    raise ValueError(
                        f"Unexpected object type: {type(message.object)}"
                    )

            elif message.user == "Tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.object["tool_call_id"],
                        "name": message.object["name"],
                        "content": message.object["content"],
                    }
                )

        return messages

    async def chat_callback(
        self, contents: str, user: str, instance: pn.chat.ChatInterface
    ):
        messages = self.custom_serializer(instance)
        response = await acompletion(
            model=MODEL,
            messages=messages,
            stream=True,
            tools=[schema(self.update_knowledge_base)],
            tool_choice="auto",
        )
        message = ""
        chunks = []
        async for chunk in response:
            content = chunk.choices[0].delta.content
            tool_calls = chunk.choices[0].delta.tool_calls

            if content is not None:
                message += content
                yield message

            if tool_calls:
                chunks.append(chunk)

        if chunks:
            rebuilt_stream = stream_chunk_builder(chunks)
            tool_call = rebuilt_stream.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            function_args_str = tool_call.function.arguments

            instance.send(
                ChatMessage(
                    visible=False,
                    **{
                        "user": "Bot",
                        "object": {
                            "tool_calls": rebuilt_stream.choices[
                                0
                            ].message.tool_calls,
                        },
                    },
                ),
                respond=False,
                avatar=None,
            )

            name_to_function_map: dict[str, t.Callable] = {
                KnowledgeGraphApp.update_knowledge_base.__name__: self.update_knowledge_base
            }

            function_to_call = name_to_function_map[function_name]
            function_args_dict = json.loads(function_args_str)
            result = function_to_call(**function_args_dict)

            instance.send(
                ChatMessage(
                    visible=False,
                    **{
                        "user": "Tool",
                        "object": {
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": result,
                        },
                    },
                ),
                respond=True,
            )

            self.graph = update_graph(result, self.graph)

    @param.depends("graph", watch=False)
    def _update_svg(self):
        return self.graph.draw()

    def __panel__(self):
        return pn.template.FastListTemplate(
            title="ChatGPT Memory (if it was cool)",
            main=[
                self._update_svg,
                self.chat_interface,
                pn.Row(
                    pn.widgets.Button(
                        name="Load Example", button_type="default"
                    ),
                    pn.layout.HSpacer(),
                ),
                pn.pane.HTML("""
                A stupid experiment by <a href="https://duarteocarmo.com" target="_blank">Duarte Carmo</a>.
                """),
            ],
            header=[],
            theme_toggle=True,
            busy_indicator=None,
            raw_css=[CSS],
            header_background="white",
            header_color="black",
            shadow=False,
            main_layout=None,
        )


app = KnowledgeGraphApp()
app.servable()
