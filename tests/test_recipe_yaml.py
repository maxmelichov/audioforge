"""Recipes must not contain duplicate mapping keys: PyYAML silently keeps the last one
(a stage-1 run trained at the wrong attention context because of a duplicated `encoder:`)."""
from pathlib import Path

import pytest
import yaml


class _Strict(yaml.SafeLoader):
    pass


def _no_dupes(loader, node, deep=False):
    keys = set()
    for k_node, _ in node.value:
        k = loader.construct_object(k_node, deep=deep)
        if k in keys:
            raise ValueError(f"duplicate key {k!r} at line {k_node.start_mark.line + 1}")
        keys.add(k)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dupes)
RECIPES = sorted(Path(__file__).parent.parent.joinpath("research", "recipes").glob("*.yaml"))


@pytest.mark.parametrize("recipe", RECIPES, ids=[r.stem for r in RECIPES])
def test_recipe_has_no_duplicate_keys(recipe):
    yaml.load(recipe.read_text(), Loader=_Strict)


def test_stage1_runs_heads_at_160ms():
    cfg = yaml.safe_load(Path("research/recipes/stage1_heads_pretrained.yaml").read_text())
    assert cfg["encoder"]["att_context_size"] == [70, 1]
