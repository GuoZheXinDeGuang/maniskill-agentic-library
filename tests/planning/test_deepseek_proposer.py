import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from mshab.experiments.granularity import build_granularity_library, contract_id
from mshab.experiments.granularity.higher_layers import (
    build_gold_graph,
    set_table_entities,
    set_table_initial_facts,
)
from mshab.experiments.planning import (
    DECOMPOSITION_SYSTEM_PROMPT,
    DEEPSEEK_API_KEY_VARIABLE,
    GRANULARITIES,
    PROMPTS,
    SCHEMA_VERSION,
    SUBGRAPH_SYSTEM_PROMPT,
    ChatReply,
    DecompositionRequest,
    DecompositionResponse,
    DeepSeekChat,
    DeepSeekProposer,
    Neighbours,
    PlanningContext,
    ProposalRejected,
    ProposalValidator,
    ProposerUnavailable,
    Rejection,
    SubgraphRequest,
    SubgraphResponse,
    extract_json_object,
)
from mshab.skills import SkillNode, SkillRelation, SkillSubgraph
from mshab.skills.schema import SchemaError


TASK = "granularity"


class RecordingChat:
    """A transport that answers from a queue and keeps every conversation it was sent."""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.conversations = []

    def __call__(self, messages):
        self.conversations.append([dict(message) for message in messages])
        if not self.replies:
            raise AssertionError("no scripted reply left")
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply


class GoldChat(RecordingChat):
    """A stand-in model that answers every request from one gold graph.

    ``overrides`` maps a sub-goal id (or ``"decomposition"``) to the replies of
    its successive turns; a turn without an override gets the gold answer.
    """

    def __init__(self, gold, overrides=None):
        super().__init__()
        self.overrides = dict(overrides or {})
        order = gold.subgoal_graph.execution_order()
        self.decomposition = DecompositionResponse(
            tuple(gold.subgoal_graph.subgoals[subgoal_id] for subgoal_id in order), "gold"
        ).as_dict()
        self.subgraphs = {
            subgoal_id: SubgraphResponse(
                gold.skill_graph.subgraph_for_subgoal(subgoal_id).unsealed_copy()
            ).as_dict()
            for subgoal_id in order
        }

    def __call__(self, messages):
        self.conversations.append([dict(message) for message in messages])
        request = json.loads(messages[1]["content"])
        turn = sum(1 for message in messages if message["role"] == "user") - 1
        key = request["subgoal"]["id"] if "subgoal" in request else "decomposition"
        turns = self.overrides.get(key, ())
        if turn < len(turns):
            return turns[turn]
        if key == "decomposition":
            return json.dumps(self.decomposition)
        return json.dumps(self.subgraphs[key])


class _Base(TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.library = build_granularity_library(Path(self.temporary_directory.name))
        self.gold = build_gold_graph("set_table_coarse", self.library)
        self.goal = self.gold.spec.goal

    def context(self, granularity="coarse"):
        return PlanningContext.initial(
            self.library,
            TASK,
            entities=set_table_entities(),
            facts=set_table_initial_facts(),
            granularity=granularity,
        )

    def bad_subgraph(self, subgoal_id="apple_delivered"):
        """A subgraph the validator refuses: its node names a contract that does not exist."""

        subgraph = SkillSubgraph(subgoal_id, TASK)
        subgraph.add_node(
            SkillNode("fly_apple", "mshab.granularity.fly.all", {"target": "013_apple"}, (subgoal_id,))
        )
        return json.dumps(SubgraphResponse(subgraph).as_dict())


class ProposerTests(_Base):
    def test_each_call_sends_the_fixed_prompt_and_the_request_json(self):
        chat = GoldChat(self.gold)
        proposer = DeepSeekProposer(chat)
        request = DecompositionRequest(TASK, self.goal, self.context())
        response = proposer.decompose(request)
        self.assertEqual(
            [subgoal.id for subgoal in response.subgoals],
            list(self.gold.subgoal_graph.execution_order()),
        )
        (conversation,) = chat.conversations
        self.assertEqual([message["role"] for message in conversation], ["system", "user"])
        self.assertEqual(conversation[0]["content"], DECOMPOSITION_SYSTEM_PROMPT)
        self.assertEqual(json.loads(conversation[1]["content"]), request.as_dict())
        exchange = proposer.exchanges[0]
        self.assertEqual((exchange.call, exchange.turn, exchange.error), ("decompose", 0, None))
        self.assertEqual(exchange.fingerprint, request.fingerprint())
        self.assertEqual(exchange.messages, tuple(conversation))

        subgoal = self.gold.subgoal_graph.subgoals["bowl_delivered"]
        subgraph_request = SubgraphRequest(
            TASK, self.goal, subgoal, request.context.contracts,
            entities=request.context.entities, facts=request.context.facts,
            neighbours=Neighbours(None, "at(013_apple,dining_table)"),
        )
        answer = proposer.plan_subgraph(subgraph_request)
        self.assertEqual(
            answer.subgraph.execution_order(),
            self.gold.skill_graph.subgraph_for_subgoal("bowl_delivered").execution_order(),
        )
        self.assertEqual(chat.conversations[1][0]["content"], SUBGRAPH_SYSTEM_PROMPT)
        self.assertEqual(json.loads(chat.conversations[1][1]["content"]), subgraph_request.as_dict())
        self.assertEqual(proposer.exchanges[1].call, "plan_subgraph")
        self.assertEqual(proposer.describe()["proposer"], "DeepSeekProposer")
        # A fresh request starts a fresh conversation; nothing is carried over.
        proposer.decompose(request)
        self.assertEqual(len(chat.conversations[2]), 2)
        self.assertEqual(len(proposer.transcript("decompose", request)), 3)

    def test_replies_are_parsed_leniently_and_then_strictly(self):
        decomposition = json.dumps(GoldChat(self.gold).decomposition)
        chat = RecordingChat(
            [
                "```json\n" + decomposition + "\n```",
                "Here is the plan:\n" + decomposition + "\nDone.",
                ChatReply(decomposition, model="deepseek-chat", usage={"total_tokens": 15}),
                "",
                "[1, 2, 3]",
                json.dumps({**json.loads(decomposition), "notes": "extra"}),
                json.dumps({"subgoals": [{"id": "a", "predicate": "x()"}, {"id": "a", "predicate": "y()"}]}),
            ]
        )
        proposer = DeepSeekProposer(chat)
        request = DecompositionRequest(TASK, self.goal, self.context())
        for _ in range(3):
            self.assertEqual(len(proposer.decompose(request).subgoals), 2)
        self.assertEqual(proposer.exchanges[2].reply.usage, {"total_tokens": 15})
        with self.assertRaisesRegex(ValueError, "empty"):
            proposer.decompose(request)
        with self.assertRaisesRegex(ValueError, "not a JSON object"):
            proposer.decompose(request)
        with self.assertRaisesRegex(SchemaError, "unknown=\\['notes'\\]"):
            proposer.decompose(request)
        with self.assertRaisesRegex(ValueError, "cannot repeat"):
            proposer.decompose(request)
        errors = [exchange.error for exchange in proposer.exchanges]
        self.assertEqual(errors[:3], [None, None, None])
        self.assertIn("empty", errors[3])
        self.assertIn("not a JSON object", errors[4])
        self.assertIn("unknown=['notes']", errors[5])
        self.assertIn("cannot repeat", errors[6])
        self.assertEqual(extract_json_object(' {"a": 1} ')["a"], 1)
        self.assertEqual(extract_json_object('text ```json\n{"a": [1, {"b": 2}]}\n``` more')["a"][1]["b"], 2)
        with self.assertRaisesRegex(ValueError, "not a JSON object"):
            extract_json_object("no braces here")

    def test_the_proposer_is_text_only(self):
        proposer = DeepSeekProposer(RecordingChat())
        context = replace(self.context(), images=("scene_0.png",))
        self.assertEqual(DecompositionRequest.from_dict(
            DecompositionRequest(TASK, self.goal, context).as_dict()).images, ("scene_0.png",))
        with self.assertRaisesRegex(ValueError, "text only"):
            proposer.decompose(DecompositionRequest(TASK, self.goal, context))
        self.assertEqual(proposer.exchanges, ())
        with self.assertRaisesRegex(TypeError, "transport options"):
            DeepSeekProposer(RecordingChat(), model="deepseek-chat")

    def test_the_transport_needs_the_package_and_a_key(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(DEEPSEEK_API_KEY_VARIABLE, None)
            try:
                import openai  # noqa: F401
            except ImportError:
                with self.assertRaisesRegex(ProposerUnavailable, "openai"):
                    DeepSeekChat()
                return
            with self.assertRaisesRegex(ProposerUnavailable, DEEPSEEK_API_KEY_VARIABLE):
                DeepSeekChat()
            chat = DeepSeekChat(api_key="sk-test", temperature=0.2)
            description = chat.describe()
            self.assertEqual(description["model"], "deepseek-chat")
            self.assertEqual(description["temperature"], 0.2)
            self.assertNotIn("sk-test", json.dumps(description))
            self.assertEqual(DeepSeekProposer(chat).describe()["transport"], "deepseek")

    def test_prompts_state_vocabulary_relations_granularities_and_schema(self):
        for prompt in (DECOMPOSITION_SYSTEM_PROMPT, SUBGRAPH_SYSTEM_PROMPT):
            self.assertIn(SCHEMA_VERSION, prompt)
            self.assertIn("JSON", prompt)
            for word in ("goal", "sub-goal", "skill node", "contract", "policy", "rejections"):
                self.assertIn(word, prompt)
        for granularity in GRANULARITIES:
            self.assertIn("- {}:".format(granularity), DECOMPOSITION_SYSTEM_PROMPT)
        for relation in SkillRelation:
            self.assertIn(relation.value, SUBGRAPH_SYSTEM_PROMPT)
        self.assertIn('"achieves": ["fridge_open"]', SUBGRAPH_SYSTEM_PROMPT)
        turn = PROMPTS.rejection_turn((Rejection("subgraph", "a", "bad node"),))
        self.assertIn('"stage": "subgraph"', turn)
        self.assertIn("bad node", turn)
        self.assertEqual(PROMPTS.decomposition, DECOMPOSITION_SYSTEM_PROMPT)


class RetryTests(_Base):
    def test_a_refused_subgraph_call_is_retried_in_its_own_conversation(self):
        chat = GoldChat(self.gold, {"apple_delivered": [self.bad_subgraph()]})
        proposer = DeepSeekProposer(chat)
        validator = ProposalValidator(self.library, TASK, retries=2)
        proposal = validator.plan(proposer, self.goal, self.context())
        self.assertEqual(proposal.skill_graph.as_dict(), self.gold.skill_graph.as_dict())
        self.assertEqual(proposal.retries, 1)
        first, second = proposal.rounds
        self.assertEqual([call.call for call in first.calls], ["decompose"] + ["plan_subgraph"] * 2)
        self.assertEqual(
            [(item.stage, item.subgoal_id) for item in first.rejections],
            [("subgraph", "apple_delivered")],
        )
        self.assertIn("unknown contract 'mshab.granularity.fly.all'", first.rejections[0].message)
        self.assertIn("unknown contract", first.calls[2].error)
        self.assertIsNotNone(first.calls[2].response)
        self.assertTrue(second.decomposition_reused)
        self.assertEqual(second.reused_subgoals, ("bowl_delivered",))
        self.assertEqual([(call.call, call.subgoal_id) for call in second.calls], [("plan_subgraph", "apple_delivered")])
        self.assertEqual(second.calls[0].request.rejections, first.rejections)
        self.assertEqual(second.calls[0].error, None)
        # The retry is a further user turn of the same conversation, after the refused answer.
        self.assertEqual(len(chat.conversations), 4)
        retry = chat.conversations[-1]
        self.assertEqual([message["role"] for message in retry], ["system", "user", "assistant", "user"])
        self.assertEqual(retry[2]["content"], self.bad_subgraph())
        self.assertIn("unknown contract 'mshab.granularity.fly.all'", retry[3]["content"])
        self.assertEqual([exchange.turn for exchange in proposer.exchanges], [0] * 3 + [1])
        # The accepted answers come from different rounds; the trace has both.
        trace = proposal.as_dict()
        self.assertEqual(len(trace["rounds"]), 2)
        self.assertEqual(trace["rounds"][0]["calls"][2]["subgoal_id"], "apple_delivered")
        self.assertEqual(trace["rounds"][1]["reused_subgoals"], list(second.reused_subgoals))
        self.assertEqual(trace["subgraphs"][1]["request"]["rejections"][0]["stage"], "subgraph")
        json.dumps(trace)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "exchanges.json"
            proposer.save_exchanges(path)
            document = json.loads(path.read_text())
        self.assertEqual(len(document), 4)
        self.assertEqual(document[-1]["turn"], 1)
        # The refused answer parsed fine; the exchange log records parse and
        # schema errors only, the validator's reason lives in the round.
        self.assertIsNone(document[2]["error"])
        self.assertEqual(json.loads(document[2]["reply"]), json.loads(self.bad_subgraph()))

    def test_retries_run_out_after_two_more_rounds(self):
        bad = self.bad_subgraph()
        chat = GoldChat(self.gold, {"apple_delivered": [bad, bad, bad]})
        with self.assertRaises(ProposalRejected) as raised:
            ProposalValidator(self.library, TASK, retries=2).plan(DeepSeekProposer(chat), self.goal, self.context())
        self.assertEqual(len(raised.exception.rounds), 3)
        self.assertEqual([len(item.calls) for item in raised.exception.rounds], [3, 1, 1])
        self.assertEqual(len(chat.conversations), 5)
        self.assertEqual(
            [message["role"] for message in chat.conversations[-1]],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )
        # Without a retry budget the first refusal is final, as before.
        chat = GoldChat(self.gold, {"apple_delivered": [bad]})
        with self.assertRaises(ProposalRejected) as raised:
            ProposalValidator(self.library, TASK).plan(DeepSeekProposer(chat), self.goal, self.context())
        self.assertEqual(len(raised.exception.rounds), 1)
        self.assertEqual(len(chat.conversations), 3)
        with self.assertRaisesRegex(ValueError, "retries cannot be negative"):
            ProposalValidator(self.library, TASK, retries=-1)

    def test_a_refused_decomposition_restarts_from_the_first_call(self):
        duplicate = json.dumps(
            {"subgoals": [{"id": "a", "predicate": "x()"}, {"id": "a", "predicate": "y()"}]}
        )
        chat = GoldChat(self.gold, {"decomposition": [duplicate]})
        proposal = ProposalValidator(self.library, TASK, retries=1).plan(
            DeepSeekProposer(chat), self.goal, self.context()
        )
        first, second = proposal.rounds
        self.assertEqual([call.call for call in first.calls], ["decompose"])
        self.assertEqual(first.rejections[0].stage, "decomposition")
        self.assertIsNone(first.calls[0].response)
        self.assertFalse(second.decomposition_reused)
        self.assertEqual(second.reused_subgoals, ())
        self.assertEqual([call.call for call in second.calls], ["decompose"] + ["plan_subgraph"] * 2)
        self.assertEqual(second.calls[0].request.rejections, first.rejections)
        self.assertEqual(
            [message["role"] for message in chat.conversations[1]],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn("cannot repeat", chat.conversations[1][3]["content"])

    def test_a_plan_stage_rejection_is_attributed_and_retried_per_sub_goal(self):
        ambiguous = SkillSubgraph("apple_delivered", TASK)
        for node_id in ("place_a", "place_b"):
            ambiguous.add_node(
                SkillNode(
                    node_id, contract_id("place"),
                    {"object": "013_apple", "destination": "dining_table"},
                    ("apple_delivered",),
                )
            )
        chat = GoldChat(
            self.gold, {"apple_delivered": [json.dumps(SubgraphResponse(ambiguous).as_dict())]}
        )
        proposal = ProposalValidator(self.library, TASK, retries=1).plan(
            DeepSeekProposer(chat), self.goal, self.context()
        )
        first, second = proposal.rounds
        self.assertEqual(
            [(item.stage, item.subgoal_id) for item in first.rejections],
            [("plan", "apple_delivered")],
        )
        self.assertIn("no FALLBACK_TO order", first.rejections[0].message)
        self.assertTrue(second.decomposition_reused)
        self.assertEqual([call.subgoal_id for call in second.calls], ["apple_delivered"])
        self.assertEqual(proposal.skill_graph.as_dict(), self.gold.skill_graph.as_dict())
