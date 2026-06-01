from config import DATASET_PATH, EDA_OUTPUT_DIR, EDA_SAMPLE_COUNT
from graphs import plot_class_distribution, plot_image_size_reduction, plot_sample_images
from load_data import load_data, print_dataset_summary
from PIL import Image


# ============================================================
# 1. SKAPA EDA-GRAFER
# ============================================================
#
# EDA används för att förstå datasetet innan modellträning. Här skapas grafer
# för klassfördelning och exempelbilder. Alla filer sparas i output/eda.

# ------------------------------------------------------------
# 1.1 Skapa output-mapp
# ------------------------------------------------------------
def create_eda_output_dir(output_dir=EDA_OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# ------------------------------------------------------------
# 1.2 Hämta originalstorlek
# ------------------------------------------------------------
def get_original_image_size(split_data):
    with Image.open(split_data.image_paths[0]) as image:
        return image.size


# ------------------------------------------------------------
# 1.3 Rita storlek före och efter reduktion
# ------------------------------------------------------------
def create_image_size_reduction_graph(split_data, output_dir):
    original_size = get_original_image_size(split_data)
    reduced_size = split_data.image_size
    save_path = output_dir / "image_size_reduction.png"
    plot_image_size_reduction(original_size, reduced_size, save_path=save_path, show=False)

    return save_path


# ------------------------------------------------------------
# 1.4 Rita klassfördelning
# ------------------------------------------------------------
def create_class_distribution_graph(split_data_list, output_dir):
    save_path = output_dir / "class_distribution.png"
    plot_class_distribution(split_data_list, save_path=save_path, show=False)

    return save_path


# ------------------------------------------------------------
# 1.5 Rita exempelbilder per datasetdel
# ------------------------------------------------------------
def create_sample_image_graphs(split_data_list, output_dir, sample_count=EDA_SAMPLE_COUNT):
    save_paths = []

    for split_data in split_data_list:
        save_path = output_dir / f"sample_images_{split_data.name}.png"
        plot_sample_images(split_data, sample_count=sample_count, save_path=save_path, show=False)
        save_paths.append(save_path)

    return save_paths


# ============================================================
# 2. KÖR EDA
# ============================================================
#
# create_eda_graphs skapar grafer från redan laddad data. run_eda är
# huvudfunktionen när EDA körs separat och laddar då datasetet själv.

# ------------------------------------------------------------
# 2.1 Skapa EDA-grafer från laddad data
# ------------------------------------------------------------
def create_eda_graphs(split_data_list, output_dir=EDA_OUTPUT_DIR, sample_count=EDA_SAMPLE_COUNT):
    output_dir = create_eda_output_dir(output_dir)

    graph_paths = [
        create_image_size_reduction_graph(split_data_list[0], output_dir),
        create_class_distribution_graph(split_data_list, output_dir),
    ]
    graph_paths.extend(create_sample_image_graphs(split_data_list, output_dir, sample_count))

    return graph_paths


# ------------------------------------------------------------
# 2.2 Kör EDA separat
# ------------------------------------------------------------
def run_eda(dataset_path=DATASET_PATH, output_dir=EDA_OUTPUT_DIR, sample_count=EDA_SAMPLE_COUNT):
    train_data, valid_data, test_data = load_data(dataset_path)
    split_data_list = [train_data, valid_data, test_data]

    print_dataset_summary(train_data, valid_data, test_data)

    return create_eda_graphs(split_data_list, output_dir, sample_count)


# ============================================================
# 3. KÖR SOM EGEN FIL
# ============================================================
#
# Filen kan köras separat när EDA-grafer ska skapas utan att starta hela
# träningspipelinen.

def main():
    graph_paths = run_eda()

    print("\nEDA-grafer skapade")
    for graph_path in graph_paths:
        print(graph_path)


if __name__ == "__main__":
    main()
