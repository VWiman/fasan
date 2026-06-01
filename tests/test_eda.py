from eda import create_eda_graphs, run_eda
from load_data import load_data


# ============================================================
# 1. TESTA EDA
# ============================================================
#
# EDA-steget ska kunna skapa alla planerade grafer i en output-mapp utan att
# använda det stora riktiga datasetet.

# ------------------------------------------------------------
# 1.1 Skapa EDA-grafer
# ------------------------------------------------------------
def test_run_eda_creates_expected_graphs(tiny_dataset, tmp_path):
    output_dir = tmp_path / "output" / "eda"

    graph_paths = run_eda(tiny_dataset, output_dir=output_dir, sample_count=2)

    assert len(graph_paths) == 5
    assert output_dir / "image_size_reduction.png" in graph_paths
    assert output_dir / "class_distribution.png" in graph_paths
    assert output_dir / "sample_images_train.png" in graph_paths
    assert output_dir / "sample_images_valid.png" in graph_paths
    assert output_dir / "sample_images_test.png" in graph_paths

    for graph_path in graph_paths:
        assert graph_path.exists()
        assert graph_path.stat().st_size > 0


# ------------------------------------------------------------
# 1.2 Skapa EDA-grafer från redan laddad data
# ------------------------------------------------------------
def test_create_eda_graphs_uses_loaded_data(tiny_dataset, tmp_path):
    output_dir = tmp_path / "output" / "eda"
    train_data, valid_data, test_data = load_data(tiny_dataset, image_size=(32, 32), batch_size=2)

    graph_paths = create_eda_graphs([train_data, valid_data, test_data], output_dir=output_dir, sample_count=2)

    assert len(graph_paths) == 5
    assert output_dir / "class_distribution.png" in graph_paths

    for graph_path in graph_paths:
        assert graph_path.exists()
