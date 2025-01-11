import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd
from docx import Document

def calculate_auc_scores(df, classes):
    """
    Calculates the AUC scores for the specified classes.

    Args:
        df (pd.DataFrame): The DataFrame containing ground truth and predictions.
        classes (list): List of class names to calculate AUC for.

    Returns:
        dict: A dictionary with class names as keys and their corresponding AUC scores as values.
    """
    auc_scores = {}
    for cls in classes:
        actual = df[cls]
        predicted = df[f'{cls}_hat']
        auc = roc_auc_score(actual, predicted)
        auc_scores[cls] = auc
    return auc_scores

def save_auc_bar_plot(auc_scores, output_file):
    """
    Creates a bar plot of AUC scores and saves it to a file.

    Args:
        auc_scores (dict): A dictionary with class names as keys and AUC scores as values.
        output_file (str): Path to save the bar plot (e.g., 'auc_scores_plot.png').
    """

    classes = list(auc_scores.keys())
    scores = list(auc_scores.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, scores, color='skyblue', edgecolor='black')
    plt.xlabel('Classes')
    plt.ylabel('AUC Score')
    plt.title('AUC Scores by Class')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


def process_file(file_path, exp_type, noise_type):
    classes = ['No Finding', 'Pleural Effusion', 'Lung Opacity', 'Atelectasis']

    df = pd.read_csv(file_path)

    y_true = df[classes].idxmax(axis=1)
    
    y_pred_raw = df[[col for col in df.columns if '_hat' in col]]
    y_pred_raw.columns = [col.replace('_hat', '') for col in y_pred_raw.columns]
    y_pred = y_pred_raw.idxmax(axis=1)
    
    # Calculate the F1 score for the entire file
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)  # Macro average F1 score

    y_true_onehot = pd.get_dummies(y_true, columns=classes)
    # auc_score = roc_auc_score(y_true_onehot, y_pred_raw, average='weighted', multi_class='ovr')

    print(file_path, "AUCs:")
    auc_scores = calculate_auc_scores(df, classes)

    # Print the AUC scores
    # for cls, auc in auc_scores.items():
    #     print(f"AUC for {cls}: {round(auc, 3)}")

    # save_auc_bar_plot(auc_scores, file_path.split('/')[-1].replace('.csv', '.png'))

    print({
        'exp_type': exp_type,
        'noise_type': noise_type,
        'f1_score': f1
    })
    return {
        'exp_type': exp_type,
        'noise_type': noise_type,
        'f1_score': f1,
    }, auc_scores

def load_and_process_data(folder_path, experiments):
    summary_data = []
    auc_scores_map = dict()
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            exp_type_part = file_name.split("_")[1]
            exp_type = 'flac' if 'flac' in exp_type_part else 'plain'
            first_part = "results14_" + exp_type_part + "_"
            second_part = file_name.replace(first_part, "")
            noise_type = 'plain'
            if "_plain" in second_part:
                noise_type = second_part.replace("_plain.csv", "")
            summary, auc_scores = process_file(file_path, exp_type, noise_type)
            if exp_type == "plain":
                auc_scores_map["densenet"+"_"+noise_type] = auc_scores
            else:
                auc_scores_map[exp_type+"_"+noise_type] = auc_scores
            summary_data.append(summary)
    print(auc_scores_map)
    return pd.DataFrame(summary_data), auc_scores_map

def plot_f1_scores(df):
    sns.set_theme(style="whitegrid")

    df['alpha_beta'] = df.apply(
        lambda row: "baseline densenet" if row['noise_type'] == 'plain' else f"{row['noise_type']}",
        axis=1
    )
    # Convert to string and then to categorical
    df['alpha_beta'] = df['alpha_beta'].astype(str)
    df['alpha_beta'] = pd.Categorical(df['alpha_beta'], 
                                       categories=sorted(df['alpha_beta'].unique()), 
                                       ordered=True)

    df['alpha_beta_index'] = df['alpha_beta'].cat.codes

    plt.figure(figsize=(14, 8))
    custom_palette = {
        "plain": "#1f77b4",  # Blue
        "flac": "#2ca02c"   # Green
    }

    bar_plot = sns.barplot(
        data=df,
        x="alpha_beta_index",
        y="f1_score",
        hue="exp_type",
        palette=custom_palette,
        ci='sd',
        capsize=.2, 
        width=0.4 
    )

    bar_plot.set_xticks(range(len(df['alpha_beta'].cat.categories)))
    bar_plot.set_xticklabels(df['alpha_beta'].cat.categories, rotation=45, ha="right")

    plt.title("F1 Scores Comparison by Noise Type - Test set without noise")
    plt.xlabel("Alpha-Beta Pair")
    plt.ylabel("F1 Score")

    for p in bar_plot.patches:
        bar_plot.annotate(f'{p.get_height():.4f}',
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='bottom', fontsize=10, color='black', 
                          xytext=(0, 5),
                          textcoords='offset points')

    plt.legend(title="Dataset Type")

    plt.ylim(0, df['f1_score'].max() + 0.05) 

    plt.tight_layout()
    plt.show()

def create_auc_table(doc_data, output_filename):
    """
    Generates a Word document with a table showcasing AUC scores from a given dictionary.

    Args:
        doc_data (dict): Dictionary containing AUC scores.
        output_filename (str): The name of the output Word file.

    Returns:
        str: Path to the generated Word document.
    """
    # Create word document
    doc = Document()
    doc.add_heading('AUC Scores', level=1)

    table = doc.add_table(rows=1, cols=5)
    table.style = 'Table Grid'

    header_cells = table.rows[0].cells
    header_cells[0].text = "Method"
    header_cells[1].text = "No Finding"
    header_cells[2].text = "Pleural Effusion"
    header_cells[3].text = "Lung Opacity"
    header_cells[4].text = "Atelectasis"

    noise_types = ["brightness_bands", "gaussian_noise", "logo", "salt_and_pepper", "plain"]
    run_types = ["flac", "densenet"]
    for noise in noise_types:
        if noise == 'plain':
            method = "Without FLAC"
            scores = doc_data["densenet_plain"]
            row_cells = table.add_row().cells
            row_cells[0].text = method
            row_cells[1].text = f"{scores['No Finding']:.3f}"
            row_cells[2].text = f"{scores['Pleural Effusion']:.3f}"
            row_cells[3].text = f"{scores['Lung Opacity']:.3f}"
            row_cells[4].text = f"{scores['Atelectasis']:.3f}"
            continue
        for run in run_types:
            scores = doc_data[run+"_"+noise]
            row_cells = table.add_row().cells
            row_cells[0].text = run.title()+" - "+noise.title()
            row_cells[1].text = f"{scores['No Finding']:.3f}"
            row_cells[2].text = f"{scores['Pleural Effusion']:.3f}"
            row_cells[3].text = f"{scores['Lung Opacity']:.3f}"
            row_cells[4].text = f"{scores['Atelectasis']:.3f}"

    # for method, scores in doc_data.items():
    #     row_cells = table.add_row().cells
    #     row_cells[0].text = method
    #     row_cells[1].text = f"{scores['No Finding']:.3f}"
    #     row_cells[2].text = f"{scores['Pleural Effusion']:.3f}"
    #     row_cells[3].text = f"{scores['Lung Opacity']:.3f}"
    #     row_cells[4].text = f"{scores['Atelectasis']:.3f}"

    doc.save(output_filename)
    return output_filename

if __name__ == "__main__":
    folder_path = "FLAC14/"
    
    data, auc_scores_map = load_and_process_data(folder_path, experiments=["brightness_bands", "gaussian_noise", "logo", "salt_and_pepper"])
    output_path = "auc_scores_for_all_noise_types_plain.docx"
    create_auc_table(auc_scores_map, output_path)

    # plot_f1_scores(data)

    # data = load_and_process_data(folder_path, experiment="brightness_bands")
    
    plot_f1_scores(data)
