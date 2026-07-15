import os
import pickle
from pathlib import Path

from typing import TypeVar
import numpy as np
import pytest
import torch
from torch import Tensor
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from .common import FIXTURES_PATH

class DEFAULT:
    pass

def pytest_addoption(parser):
    parser.addoption("--snapshot-exact", action="store_true", help="Use exact matching standards for snapshot matching")

_A = TypeVar("_A", np.ndarray, Tensor)


def _canonicalize_array[A: (np.ndarray, Tensor)](arr: A) -> np.ndarray:
    if isinstance(arr, Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


class NumpySnapshot[A: (np.ndarray, Tensor)]:
    """Snapshot testing utility for NumPy arrays using .npz format."""

    def __init__(
        self,
        snapshot_dir: str = "tests/_snapshots",
        default_force_update: bool = False,
        always_match_exact: bool = False,
        default_test_name: str | None = None,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.default_force_update = default_force_update
        self.always_match_exact = always_match_exact
        self.default_test_name = default_test_name

    def _get_snapshot_path(self, test_name: str) -> Path:
        """Get the path to the snapshot file."""
        return self.snapshot_dir / f"{test_name}.npz"

    def assert_match(
        self,
        actual: A | dict[str, A],
        rtol: float = 1e-4,
        atol: float = 1e-2,
        test_name: str | type[DEFAULT] = DEFAULT,
        force_update: bool | type[DEFAULT] = DEFAULT,
    ):
        """
        Assert that the actual array(s) matches the snapshot.

        Args:
            actual: Single NumPy array or dictionary of named arrays
            test_name: The name of the test (used for the snapshot file)
            update: If True, update the snapshot instead of comparing
        """
        if force_update is DEFAULT:
            force_update = self.default_force_update
        if self.always_match_exact:
            rtol = atol = 0
        if test_name is DEFAULT:
            assert self.default_test_name is not None, "Test name must be provided or set as default"
            test_name = self.default_test_name

        snapshot_path = self._get_snapshot_path(test_name)

        # Convert single array to dictionary for consistent handling
        arrays_dict = actual if isinstance(actual, dict) else {"array": actual}
        arrays_dict = {k: _canonicalize_array(v) for k, v in arrays_dict.items()}

        # Load the snapshot
        expected_arrays = dict(np.load(snapshot_path))

        # Verify all expected arrays are present
        missing_keys = set(arrays_dict.keys()) - set(expected_arrays.keys())
        if missing_keys:
            raise AssertionError(f"Keys {missing_keys} not found in snapshot for {test_name}")

        # Verify all actual arrays are expected
        extra_keys = set(expected_arrays.keys()) - set(arrays_dict.keys())
        if extra_keys:
            raise AssertionError(f"Snapshot contains extra keys {extra_keys} for {test_name}")

        # Compare all arrays
        for key in arrays_dict:
            np.testing.assert_allclose(
                _canonicalize_array(arrays_dict[key]),
                expected_arrays[key],
                rtol=rtol,
                atol=atol,
                err_msg=f"Array '{key}' does not match snapshot for {test_name}",
            )


class Snapshot[A: (np.ndarray, Tensor)]:
    def __init__(
        self,
        snapshot_dir: str = "tests/_snapshots",
        default_force_update: bool = False,
        default_test_name: str | None = None,
    ):
        """
        Snapshot for arbitrary data types, saved as pickle files.
        """
        self.snapshot_dir = Path(snapshot_dir)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.default_force_update = default_force_update
        self.default_test_name = default_test_name

    def _get_snapshot_path(self, test_name: str) -> Path:
        return self.snapshot_dir / f"{test_name}.pkl"

    def assert_match(
        self,
        actual: A | dict[str, A],
        test_name: str | type[DEFAULT] = DEFAULT,
        force_update: bool | type[DEFAULT] = DEFAULT,
    ):
        """
        Assert that the actual data matches the snapshot.
        Args:
            actual: Single object or dictionary of named objects
            test_name: The name of the test (used for the snapshot file)
            force_update: If True, update the snapshot instead of comparing
        """

        if force_update is DEFAULT:
            force_update = self.default_force_update
        if test_name is DEFAULT:
            assert self.default_test_name is not None, "Test name must be provided or set as default"
            test_name = self.default_test_name

        snapshot_path = self._get_snapshot_path(test_name)

        # Load the snapshot
        with open(snapshot_path, "rb") as f:
            expected_data = pickle.load(f)

        if isinstance(actual, dict):
            for key in actual:
                if key not in expected_data:
                    raise AssertionError(f"Key '{key}' not found in snapshot for {test_name}")
                assert actual[key] == expected_data[key], (
                    f"Data for key '{key}' does not match snapshot for {test_name}"
                )
        else:
            assert actual == expected_data, f"Data does not match snapshot for {test_name}"


@pytest.fixture
def snapshot(request):
    """
    Fixture providing snapshot testing functionality.

    Usage:
        def test_my_function(snapshot):
            result = my_function()
            snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    # Create the snapshot handler with default settings
    snapshot_handler = Snapshot()

    # Patch the assert_match method to include the update flag by default
    original_assert_match = snapshot_handler.assert_match

    def patched_assert_match(actual, test_name=None, force_update=force_update):
        # If test_name is not provided, use the test function name
        if test_name is None:
            test_name = request.node.name
        return original_assert_match(actual, test_name=test_name, force_update=force_update)

    snapshot_handler.assert_match = patched_assert_match

    # # Create the snapshot handler with default settings
    # snapshot_handler = Snapshot(default_force_update=force_update, default_test_name=request.node.name)

    return snapshot_handler


# Fixture that can be used in all tests
@pytest.fixture
def numpy_snapshot(request):
    """
    Fixture providing numpy snapshot testing functionality.

    Usage:
        def test_my_function(numpy_snapshot):
            result = my_function()
            numpy_snapshot.assert_match(result, "my_test_name")
    """
    force_update = False

    match_exact = request.config.getoption("--snapshot-exact", default=False)

    # Create the snapshot handler with default settings
    snapshot = NumpySnapshot()

    # Patch the assert_match method to include the update flag by default
    original_assert_match = snapshot.assert_match

    def patched_assert_match(actual, test_name=None, force_update=force_update, rtol=1e-4, atol=1e-2):
        # If test_name is not provided, use the test function name
        if test_name is None:
            test_name = request.node.name
        if match_exact:
            rtol = atol = 0
        return original_assert_match(actual, test_name=test_name, force_update=force_update, rtol=rtol, atol=atol)

    snapshot.assert_match = patched_assert_match
    
    # # Create the snapshot handler with default settings
    # snapshot = NumpySnapshot(
    #     default_force_update=force_update, always_match_exact=match_exact, default_test_name=request.node.name
    # )

    return snapshot


@pytest.fixture
def ts_state_dict(request):
    import json

    from .common import FIXTURES_PATH

    state_dict = torch.load(FIXTURES_PATH / "ts_tests" / "model.pt", map_location="cpu")
    config = json.load(open(FIXTURES_PATH / "ts_tests" / "model_config.json"))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict, config


# Model parameters used for model fixture


@pytest.fixture
def n_layers():
    return 3


# For assignment 1/2
@pytest.fixture
def vocab_size():
    return 10_000


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def n_queries():
    return 12


@pytest.fixture
def n_keys():
    return 16


@pytest.fixture
def n_heads():
    return 4


@pytest.fixture
def d_head():
    return 16


@pytest.fixture
def d_model(n_heads, d_head):
    return n_heads * d_head


@pytest.fixture
def d_ff():
    return 128


@pytest.fixture
def q(batch_size, n_queries, d_model):
    torch.manual_seed(1)
    return torch.randn(batch_size, n_queries, d_model)


@pytest.fixture
def k(batch_size, n_keys, d_model):
    torch.manual_seed(2)
    return torch.randn(batch_size, n_keys, d_model)


@pytest.fixture
def v(batch_size, n_keys, d_model):
    torch.manual_seed(3)
    return torch.randn(batch_size, n_keys, d_model)


@pytest.fixture
def in_embeddings(batch_size, n_queries, d_model):
    torch.manual_seed(4)
    return torch.randn(batch_size, n_queries, d_model)


@pytest.fixture
def mask(batch_size, n_queries, n_keys):
    torch.manual_seed(5)
    return torch.randn(batch_size, n_queries, n_keys) > 0.5


@pytest.fixture
def in_indices(batch_size, n_queries):
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (batch_size, n_queries))


@pytest.fixture
def theta():
    return 10000.0


@pytest.fixture
def pos_ids(n_queries):
    return torch.arange(0, n_queries)


# # Example usage:
# def test_single_array(numpy_snapshot):
#     # Sample function that produces a numpy array
#     def my_function():
#         return np.array([[1.0, 2.0], [3.0, 4.0001]])

#     result = my_function()

#     # Just provide the result - the test name will be inferred
#     numpy_snapshot.assert_match(result)


# def test_multiple_arrays(numpy_snapshot):
#     # Function that produces multiple arrays
#     def my_function():
#         return {
#             "weights": np.array([0.1, 0.2, 0.3]),
#             "biases": np.array([0.01, 0.02]),
#             "gradients": np.array([[0.001, 0.002], [0.003, 0.004]])
#         }

#     results = my_function()

#     # Test with explicit name and custom tolerances
#     # custom_snapshot = NumpySnapshot()
#     numpy_snapshot.assert_match(
#         results,
#         "my_special_test",
#         rtol=1e-4,
#         atol=1e-5,
#     )

# def test_state_dict(ts_state_dict):
#     print(ts_state_dict)



@pytest.fixture
def prompt_strs():
    return [
        "Hello, world!",
        "This is a test.",
        "This is another test.",
    ]


@pytest.fixture
def output_strs():
    return [
        "Hello, world!",
        "This is a test.",
        "This is another test.",
    ]


@pytest.fixture
def model_id():
    return FIXTURES_PATH / "tiny-gpt2"


@pytest.fixture
def tokenizer():
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "Hello": 3,
        "world": 4,
        "This": 5,
        "is": 6,
        "a": 7,
        "test": 8,
        "another": 9,
        "Question": 10,
        "Answer": 11,
        "Instruction": 12,
        "Response": 13,
        "###": 14,
    }
    word_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    word_tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=word_tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


@pytest.fixture
def model(model_id):
    return AutoModelForCausalLM.from_pretrained(model_id)


@pytest.fixture
def tiny_train_model(tokenizer):
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=16,
        n_ctx=16,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_inner=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.train()
    return model.cpu()


@pytest.fixture
def reward_fn():
    def dummy_reward_fn(response, ground_truth):
        # Use SHA-256 which is deterministic
        response_hash = int(hashlib.sha256(response.encode()).hexdigest(), 16)
        reward = (response_hash % 10) / 10.0
        return {
            "reward": reward,
            "format_reward": reward,
            "answer_reward": reward,
        }

    return dummy_reward_fn


@pytest.fixture
def num_rollout_responses():
    return 8


@pytest.fixture
def group_size(num_rollout_responses):
    return int(num_rollout_responses / 2)


@pytest.fixture
def rollout_responses(num_rollout_responses):
    return [f"hmm I think ths answer is {i}" for i in range(num_rollout_responses)]


@pytest.fixture
def repeated_ground_truths(num_rollout_responses):
    return ["42"] * num_rollout_responses


@pytest.fixture
def advantage_eps():
    return 1e-6


@pytest.fixture
def seq_length():
    return 10

# For assignment 5
@pytest.fixture
def vocab_size2():
    return 100

@pytest.fixture
def batch_size2():
    return 2


@pytest.fixture
def logits(batch_size2, seq_length, vocab_size2):
    torch.manual_seed(42)
    return torch.randn(size=(batch_size2, seq_length, vocab_size2))


@pytest.fixture
def input_ids(batch_size2, seq_length, vocab_size2):
    torch.manual_seed(42)
    return torch.randint(0, vocab_size2, size=(batch_size2, seq_length))


@pytest.fixture
def labels(input_ids):
    last_tokens = torch.zeros(size=(input_ids.shape[0], 1), dtype=input_ids.dtype)
    return torch.cat([input_ids[:, 1:], last_tokens], dim=1)


@pytest.fixture
def raw_rewards_or_advantages(batch_size2):
    torch.manual_seed(42)
    return torch.rand(size=(batch_size2, 1))


@pytest.fixture
def policy_log_probs(batch_size2, seq_length):
    torch.manual_seed(42)
    return torch.randn(size=(batch_size2, seq_length))


@pytest.fixture
def old_log_probs(policy_log_probs):
    torch.manual_seed(42)
    return policy_log_probs + torch.randn_like(policy_log_probs)


@pytest.fixture
def advantages(raw_rewards_or_advantages):
    return raw_rewards_or_advantages - torch.mean(raw_rewards_or_advantages, dim=0)


@pytest.fixture
def raw_rewards(raw_rewards_or_advantages):
    return raw_rewards_or_advantages


@pytest.fixture
def tensor(logits):
    return logits


@pytest.fixture
def mask2(tensor):
    torch.manual_seed(42)
    return torch.rand_like(tensor) > 0.5


@pytest.fixture
def response_mask(policy_log_probs):
    torch.manual_seed(42)
    return torch.rand_like(policy_log_probs) > 0.5


@pytest.fixture
def gradient_accumulation_steps():
    return 2


@pytest.fixture
def cliprange():
    return 0.1


@pytest.fixture
def normalize_constant():
    return 42.0
