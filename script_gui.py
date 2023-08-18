import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from PIL import Image, ImageTk


# Function to fill missing values with user choice
def fill_missing_values_with_user_choice(df, output_text):
    """
    Fill missing values in a DataFrame by allowing the user to choose 
    the statistic (mean, median, mode) to use for each column with missing
    values. Skips columns with all values present or non-numerical data.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        output_text (tk.Text): Tkinter Text widget to display output.

    Returns:
        None. The function modifies the DataFrame in place by filling 
        missing values.
    """
    for column_name in df.columns:
        if df[column_name].dtypes != "int64" and df[column_name].dtypes != "float64":
            output_text.insert(
                tk.END, f"Skipping column '{column_name}' with non-numeric data.\n"
            )
            output_text.insert(tk.END, "-" * 50 + "\n")
            continue

        column_data = df[column_name]
        column_data_without_missing = column_data.dropna()

        if column_data.isna().any():
            # Calculate mean, median, and mode
            mean_val = column_data_without_missing.mean()
            median_val = column_data_without_missing.median()
            mode_array = stats.mode(column_data_without_missing)
            mode_val = (
                mode_array[0][0]
                if isinstance(mode_array, np.ndarray) and len(mode_array[0]) > 0
                else None
            )

            output_text.insert(tk.END, f"Column: {column_name}\n")
            output_text.insert(tk.END, f"Mean: {mean_val:.2f}\n")
            output_text.insert(tk.END, f"Median: {median_val:.2f}\n")
            output_text.insert(tk.END, f"Mode: {mode_val}\n")

            # Ask user for choice of statistic
            # while True:
            #     choice = simpledialog.askstring("Input", f"Choose the statistic to fill missing values for '{column_name}' (mean, median, mode, none):").lower().strip()
            #     if choice is None:
            #         choice = "none"
            #     if choice in ['mean', 'median', 'mode', 'none']:
            #         break
            #     else:
            #         output_text.insert(tk.END, "Invalid choice.\n")
            choice = "mean"

            # Fill missing values based on user's choice
            if choice == "mean":
                fill_value = mean_val
            elif choice == "median":
                fill_value = median_val
            elif choice == "mode":
                fill_value = mode_val
            elif choice == "none":
                output_text.insert(
                    tk.END, f"No missing values filled for '{column_name}'.\n"
                )
                output_text.insert(tk.END, "=" * 50 + "\n")
                continue

            df[column_name].fillna(fill_value, inplace=True)
            output_text.insert(
                tk.END,
                f"Missing values in column '{column_name}' filled with {choice}.\n",
            )
            output_text.insert(tk.END, "=" * 50 + "\n")
        else:
            output_text.insert(
                tk.END, f"Skipping column '{column_name}' with no missing data.\n"
            )
            output_text.insert(tk.END, "-" * 50 + "\n")
            continue

    output_text.insert(tk.END, "Finished replacing numerical values\n")
    output_text.insert(tk.END, "-" * 50 + "\n")
    output_text.see(tk.END)


def fill_missing_values_with_median(df, output_text):
    """
    Fill missing values in a DataFrame using the median for each column with missing values.
    Skips columns with all values present or non-numerical data.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None. The function modifies the DataFrame in place by filling missing values with medians.
    """
    for column_name in df.columns:
        if df[column_name].dtypes != "int64" and df[column_name].dtypes != "float64":
            output_text.insert(tk.END, f"\nSkipping column '{column_name}' with non-numeric data.\n")
            output_text.insert(tk.END, "\n")
            output_text.insert(tk.END, "=" * 75 + "\n")
            continue

        column_data = df[column_name]

        if column_data.isna().any():
            median_val = column_data.median()
            
            output_text.insert(tk.END, f"\nColumn: {column_name}\n")
            output_text.insert(tk.END, f"Median: {median_val:.2f}\n")
            
            df[column_name].fillna(median_val, inplace=True)
            output_text.insert(tk.END, f"Missing values in column '{column_name}' filled with median.\n")
            output_text.insert(tk.END, "\n")
            output_text.insert(tk.END, "=" * 75 + "\n")

        else:
            output_text.insert(tk.END, f"\nSkipping column '{column_name}' with no missing data.\n")
            output_text.insert(tk.END, "\n")
            output_text.insert(tk.END, "=" * 75 + "\n")
            continue
    
    output_text.insert(tk.END, "Finished replacing numerical values.\n")


def load_data(output_text):
    global tx_antenna_dab, tx_params_dab
    file_path_antenna = filedialog.askopenfilename(
        title="Select Antenna Data CSV file", filetypes=[("CSV Files", "*.csv")]
    )
    if file_path_antenna:
        tx_antenna_dab = pd.read_csv(file_path_antenna)
        fill_missing_values_with_median(tx_antenna_dab, output_text)

    file_path_params = filedialog.askopenfilename(
        title="Select Params Data CSV file", filetypes=[("CSV Files", "*.csv")]
    )
    if file_path_params:
        tx_params_dab = pd.read_csv(file_path_params, encoding="ISO-8859-1")
        fill_missing_values_with_median(tx_params_dab, output_text)

    # Merge the two DataFrames
    tx_antenna_dab = tx_antenna_dab.merge(tx_params_dab, how="left", on="id")

    # Convert 'Date' column to datetime type
    tx_antenna_dab["Date"] = pd.to_datetime(tx_antenna_dab["Date"])


def display_statistics(output_text, label, mean_val, median_val, mode_val):
    output_text.insert(tk.END, "\n")
    output_text.insert(tk.END, "=" * 75 + "\n")
    output_text.insert(tk.END, f"{label}:\n")
    output_text.insert(tk.END, f"\tMean:\t {mean_val:.2f}\n")
    output_text.insert(tk.END, f"\tMedian:\t {median_val:.2f}\n")
    output_text.insert(tk.END, f"\tMode:\t {mode_val}\n")
    output_text.insert(tk.END, "=" * 75 + "\n")


def filter_data(output_text):
    global tx_antenna_dab, dab_multiplex_data
    # Filter out rows with specific 'NGR' values
    ngrs_to_exclude = ["NZ02553847", "SE213515", "NT05399374", "NT25265908"]
    tx_antenna_dab = tx_antenna_dab[~tx_antenna_dab["NGR"].isin(ngrs_to_exclude)]

    # Extract the DAB multiplexes from 'EID' into new columns
    dab_multiplexes = ["C18A", "C18F", "C188"]
    for multiplex in dab_multiplexes:
        tx_antenna_dab[multiplex] = (
            tx_antenna_dab["EID"].str.contains(multiplex).astype(int)
        )

    # Rename columns and convert 'In-Use Ae Ht' and 'In-Use ERP Total' to numeric values
    tx_antenna_dab.rename(
        columns={"In-Use Ae Ht": "Aerial height (m)", "In-Use ERP Total": "Power (kW)"},
        inplace=True,
    )

    # Convert columns to string and then perform 'str.replace' and 'astype' operations
    tx_antenna_dab["Aerial height (m)"] = (
        tx_antenna_dab["Aerial height (m)"]
        .astype(str)
        .str.replace(",", "")
        .astype(float)
    )
    tx_antenna_dab["Power (kW)"] = (
        tx_antenna_dab["Power (kW)"].astype(str).str.replace(",", "").astype(float)
    )

    # Filter the data for each multiplex value (C18A, C18F, C188) within the filtered data
    c18a_data = tx_antenna_dab[tx_antenna_dab['C18A'] == 1]
    c18f_data = tx_antenna_dab[tx_antenna_dab['C18F'] == 1]
    c188_data = tx_antenna_dab[tx_antenna_dab['C188'] == 1]

    # Prepare the data for DAB multiplexes
    column_data = ['Site', 'Freq.', 'Block', 'Serv Label1 ', 'Serv Label2 ', 'Serv Label3 ', 'Serv Label4 ','Serv Label10 ']

    # Separate the categories
    c18a_visualisation_data = c18a_data[column_data]
    c18f_visualisation_data = c18f_data[column_data]
    c188_visualisation_data = c188_data[column_data]

    # Concatenate the categories into one dataframe for visualisation
    dab_multiplex_data = pd.concat([c18a_visualisation_data,
                                    c18f_visualisation_data,
                                    c188_visualisation_data])
    
    # Filter the data for 'Site Height' more than 75 and 'Date' from 2001 onwards
    filtered_c18a_data = c18a_data[(c18a_data['Site Height'] > 75) & (c18a_data['Date'] >= '2001-01-01')]
    filtered_c18f_data = c18f_data[(c18f_data['Site Height'] > 75) & (c18f_data['Date'] >= '2001-01-01')]
    filtered_c188_data = c188_data[(c188_data['Site Height'] > 75) & (c188_data['Date'] >= '2001-01-01')]


    # Extract the 'Power (kW)' values for each multiplex within the filtered data
    c18a_power_values = filtered_c18a_data['Power (kW)'].dropna().astype(float)
    c18f_power_values = filtered_c18f_data['Power (kW)'].dropna().astype(float)
    c188_power_values = filtered_c188_data['Power (kW)'].dropna().astype(float)

    # Calculate the mean, median, and mode for each set of 'Power (kW)' values
    mean_c18a = c18a_power_values.mean()
    median_c18a = c18a_power_values.median()
    mode_c18a = stats.mode(c18a_power_values).mode[0]

    mean_c18f = c18f_power_values.mean()
    median_c18f = c18f_power_values.median()
    mode_c18f = stats.mode(c18f_power_values).mode[0]

    mean_c188 = c188_power_values.mean()
    median_c188 = c188_power_values.median()
    mode_c188 = stats.mode(c188_power_values).mode[0]

    # Display the results
    display_statistics(output_text, "C18A Data", mean_c18a, median_c18a, mode_c18a)
    display_statistics(output_text, "C18F Data", mean_c18f, median_c18f, mode_c18f)
    display_statistics(output_text, "C188 Data", mean_c188, median_c188, mode_c188)


def visualize_data(output_text):
    # Create a list of PNG file names
    image_files = []

    # Create a directory to save the images
    images_dir = "visualization_images"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Create and save each graph as a separate image
    sns.set(style="whitegrid")  # Set the style for all plots

    # Specify the number of top locations you want to display
    n_top_locations = 20

    # Get the top N locations and their counts
    top_locations = dab_multiplex_data["Site"].value_counts().head(n_top_locations)

    # Create a barchart
    plt.figure(figsize=(10, 6))
    sns.countplot(data=dab_multiplex_data, x="Site", order=top_locations.index)
    plt.xticks(rotation=90)
    plt.xlabel("Site")
    plt.ylabel("Frequency Count")
    plt.title(f"Top {n_top_locations} Frequency Count of Different Locations")
    plt.tight_layout()

    # Save as a PNG
    plt.savefig(os.path.join(images_dir,
                             f"top_{n_top_locations}_frequency_count.png"))

    # Add it to the image list
    image_files.append(
        f"top_{n_top_locations}_frequency_count.png",
    )

    # Set up the figure and axes
    plt.figure(figsize=(12, 6))

    # Create a histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data=dab_multiplex_data, x="Freq.", bins=20, kde=False)
    plt.xlabel("Frequency")
    plt.ylabel("Frequency Count")
    plt.title("Frequency Distribution (Histogram)")

    # Create a kernel density plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=dab_multiplex_data, x="Freq.")
    plt.xlabel("Frequency")
    plt.ylabel("Density")
    plt.title("Frequency Distribution (Kernel Density Plot)")

    # Adjust layout and show the plot
    plt.tight_layout()

    # Save as a PNG
    plt.savefig(os.path.join(images_dir, "frequency_distribution.png"))

    image_files.append("frequency_distribution.png")

    # Set up the figure and axes
    plt.figure(figsize=(10, 6))

    # Define custom colors for each block value
    colours = sns.color_palette("Set2")

    # Create a histogram for block distribution with custom colors using the palette parameter
    sns.histplot(
        data=dab_multiplex_data,
        x="Block",
        bins=len(dab_multiplex_data["Block"].unique()),
        kde=False,
        hue="Block",
        palette=colours,
    )
    plt.xlabel("Block")
    plt.ylabel("Block Count")
    plt.title("Block Distribution")

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()

    # Save as a PNG
    plt.savefig(os.path.join(images_dir, "block_distribution.png"))

    image_files.append("block_distribution.png")

    # Set up the figure and axes
    plt.figure(figsize=(12, 6))

    # Gather the unique service labels
    service_labels = dab_multiplex_data[
        [
            "Serv Label1 ",
            "Serv Label2 ",
            "Serv Label3 ",
            "Serv Label4 ",
            "Serv Label10 ",
        ]
    ].values.ravel()

    # Calculate the counts of each unique service label
    service_label_counts = pd.Series(service_labels).value_counts()

    # Create a bar plot for unique service label counts
    sns.barplot(
        x=service_label_counts.index, y=service_label_counts.values, palette="viridis"
    )
    plt.xlabel("Service Label")
    plt.ylabel("Count")
    plt.title("Service Label Counts")
    plt.xticks(rotation=55, ha="right")

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "service_label_counts.png"))

    image_files.append("service_label_counts.png")

    # Set up the figure and axes
    plt.figure(figsize=(12, 6))

    # Calculate the min and max frequency value
    min_frequency = dab_multiplex_data["Freq."].min()
    max_frequency = dab_multiplex_data["Freq."].max()

    flatui_palette = sns.color_palette(
        [
            "#9b59b6",
            "#3498db",
            "#95a5a6",
            "#e74c3c",
            "#34495e",
            "#2ecc71",
            "#1abc9c",
            "#d35400",
            "#f39c12",
            "#c0392b",
            "#8e44ad",
            "#2980b9",
            "#27ae60",
            "#e67e22",
            "#bdc3c7",
            "#16a085",
            "#f1c40f",
            "#7f8c8d",
            "#d35400",
            "#2980b9",
        ]
    )

    # Create a box plot for distribution of frequencies by Service Label
    sns.barplot(data=dab_multiplex_data, x="Site", y="Freq.", palette=flatui_palette)
    plt.xlabel("Site")
    plt.ylabel("Frequency")
    plt.title("Categorical Box Plot: Distribution of Frequencies by Site")

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Scale the y axis
    plt.ylim((min_frequency - 10), (max_frequency + 10))
    plt.tight_layout()

    # Save as a PNG
    plt.savefig(os.path.join(images_dir, "service_label_counts.png"))

    image_files.append("service_label_counts.png")

    # Calculate the value counts for each block
    block_value_counts = dab_multiplex_data["Block"].value_counts()

    # Set a threshold for the percentage value to show
    percentage_threshold = 5  # Only show percentage for slices above 5%

    # Filter blocks that meet the threshold
    filtered_blocks = block_value_counts[block_value_counts > percentage_threshold]

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        filtered_blocks,
        labels=filtered_blocks.index,
        autopct=lambda p: "{:.1f}%".format(p) if p > percentage_threshold else "",
        startangle=140,
    )
    plt.title(f"Pie Chart: Distribution of Blocks (Threshold: {percentage_threshold}%)")
    plt.tight_layout()

    # Save as a PNG
    plt.savefig(os.path.join(images_dir, "distribution_of_blocks_pie.png"))

    image_files.append("distribution_of_blocks_pie.png")

    # Calculate the value counts for each block
    freq_value_counts = dab_multiplex_data["Freq."].value_counts()

    # Set a threshold for the percentage value to show
    percentage_threshold = 5  # Only show percentage for slices above 5%

    # Filter blocks that meet the threshold
    filtered_freq = freq_value_counts[freq_value_counts > percentage_threshold]

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        filtered_freq,
        labels=filtered_freq.index,
        autopct=lambda p: "{:.1f}%".format(p) if p > percentage_threshold else "",
        startangle=140,
    )
    plt.title(
        f"Pie Chart: Distribution of Frequency (Threshold: {percentage_threshold}%)"
    )
    plt.tight_layout()

    # Save as a PNG
    plt.savefig(os.path.join(images_dir, "distribution_of_frequency_pie.png"))

    image_files.append("distribution_of_frequency_pie.png")

    # Create a copy of the original DataFrame to avoid modifying the original data
    dab_multiplex_data_encoded = dab_multiplex_data.copy()

    # Apply label encoding to each categorical column
    label_encoder = LabelEncoder()
    for column in dab_multiplex_data_encoded.columns:
        if dab_multiplex_data_encoded[column].dtype == "object":
            dab_multiplex_data_encoded[column] = label_encoder.fit_transform(
                dab_multiplex_data_encoded[column].astype(str)
            )
        # Apply label encoding to each categorical column
        label_encoder = LabelEncoder()
        for column in dab_multiplex_data_encoded.columns:
            if dab_multiplex_data_encoded[column].dtype == "object":
                dab_multiplex_data_encoded[column] = label_encoder.fit_transform(
                    dab_multiplex_data_encoded[column].astype(str)
                )

    correlation_matrix = dab_multiplex_data_encoded.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, "correlation_matrix_heatmap.png"))

    image_files.append("correlation_matrix_heatmap.png")

    # Create a new window to display the images
    graph_window = tk.Toplevel()
    graph_window.title("Visualize Data")

    # Create a canvas for the images and attach a scrollbar
    canvas = tk.Canvas(graph_window)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(graph_window, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a frame inside the canvas to hold the images
    image_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=image_frame, anchor=tk.NW)

    # Load images using PIL and display them in the frame
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)

        # Load the image using PIL
        img_pil = Image.open(image_path)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Display the image in a Label widget
        label = tk.Label(image_frame, image=img_tk)
        label.image = img_tk  # Keep a reference to the image to prevent it from being garbage collected
        label.pack()

    # Configure the canvas scrolling region
    canvas.update_idletasks()
    canvas.config(scrollregion=canvas.bbox(tk.ALL))


def create_gui():
    root = tk.Tk()
    root.title("Data Analysis GUI")

    output_text = tk.Text(root, wrap=tk.WORD)
    output_text.pack()

    load_button = tk.Button(
        root, text="Load and Clean Data", command=lambda: load_data(output_text)
    )
    load_button.pack()

    filter_button = tk.Button(
        root,
        text="Filter Data and Show Statistics",
        command=lambda: filter_data(
            output_text,
        ),
    )
    filter_button.pack()

    visualize_button = tk.Button(
        root, text="Visualize Data", command=lambda: visualize_data(output_text)
    )
    visualize_button.pack()

    root.mainloop()


if __name__ == "__main__":
    create_gui()
